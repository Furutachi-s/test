import os
import streamlit as st
import fitz  # PyMuPDFとしてインポート
from openai import AzureOpenAI
import pandas as pd
from io import BytesIO
import json
import tiktoken
import base64
from pymupdf4llm import to_markdown  # PyMuPDF4LLMのto_markdown関数をインポート
import time  # 処理時間計測のため追加
import re

# 環境変数からAPIキーとエンドポイントを取得
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

# チャット履歴を保存するリスト
chat_history = []

# チャット形式でOpenAIに問い合わせる関数
def check_text_with_openai(text, page_num):
    session_message = []
    system_prompt = "You are a helpful assistant."
    session_message.append({"role": "system", "content": system_prompt})

    prompt = f"""
あなたには建設機械のマニュアルをチェックしてもらいます（該当ページ: {page_num}）。
以下の項目についてチェックし、指摘してください: 
(1) 誤字
(2) 文脈的に、記載内容が明確に間違っている場合
(3) 文法が致命的に間違っていて、文章として意味が通らない場合

ただし、以下の項目については指摘しないこと:
(1) 偽陽性を避けるため、判断に迷った場合は指摘しない
(2) 不自然な空白、半角スペースはPDF抽出時の仕様のため、指摘しない

重要なポイント
・偽陽性を避けるため、判断に迷った場合は指摘しないでください。
・各指摘については、以下の形式でJSONとして返し、JSON以外の文字列を一切含めないでください。コードブロックや追加の説明も含めないでください。
- "page": 該当ページ番号
- "category": 指摘のカテゴリ（例: 誤字、文法、文脈）
- "reason": 指摘した理由（簡潔に記述）
- "error_location": 指摘箇所
- "context": 周辺テキスト

以下の文章についてチェックを行ってください：
\n\n{text}
"""
    session_message.append({"role": "user", "content": prompt})

    # チャット履歴に追加
    chat_history.append({"page_num": page_num, "messages": session_message})

    for attempt in range(3):  # 最大3回リトライ
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                seed=42,
                temperature=0,
                max_tokens=2000,
                messages=session_message
            )
            # チャット履歴にAIの応答を追加
            chat_history[-1]["response"] = response.choices[0].message.content
            return response.choices[0].message.content
        except Exception as e:
            st.write(f"エラーが発生しました。再試行します... ({e})")
            time.sleep(5)
    st.write("エラーが続いたため、このチャンクをスキップします。")
    # チャット履歴にエラー情報を追加
    chat_history[-1]["response"] = f"エラー: {e}"
    return ""

# 画像をBase64形式でエンコードして埋め込む関数
def encode_image_to_base64(image):
    image_stream = BytesIO()
    image.save(image_stream, format="PNG")  # PNG形式で保存
    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return f"![Embedded Image](data:image/png;base64,{image_base64})"

# PyMuPDF4LLMを使用してPDFを解析し、ページごとにテキストを抽出する関数
def extract_text_with_pymupdf4llm(pdf_document):
    page_texts = []
    for page_num in range(len(pdf_document)):
        try:
            st.write(f"ページ {page_num + 1} を処理しています...")
            # 修正ポイント: ドキュメント全体を渡し、pagesパラメータでページを指定
            markdown_text = to_markdown(pdf_document, pages=[page_num])
            page_texts.append((page_num + 1, markdown_text))  # ページ番号とテキストをタプルで保存
        except Exception as e:
            st.write(f"ページ {page_num + 1} の処理中にエラーが発生しました: {e}")
            page_texts.append((page_num + 1, ""))  # 空のテキストを追加して処理を続行
    return page_texts

# テキストをトークン数に基づいてチャンクに分割する関数
def split_text_into_chunks_by_page(page_texts, chunk_size=2000, chunk_overlap=200):
    encoding = tiktoken.encoding_for_model('gpt-4')
    page_chunks = []
    
    for page_num, text in page_texts:
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            page_chunks.append((page_num, chunk_text))  # ページ番号とチャンクを対応させる
    
    return page_chunks

# チャンクが意図通り分割されているか確認するデバッグ表示
def display_chunks_debug(page_chunks):
    st.subheader("チャンク内容のデバッグ表示（ページごと）")
    for page_num, chunk_text in page_chunks:
        st.write(f"ページ {page_num} の内容:")
        st.write(chunk_text[:100] + "...")  # 最初の100文字を表示
        st.write("---")

# チェック結果をパースしてDataFrameに変換する関数
def parse_json_results_to_dataframe(results, page_num):
    issues = []
    results = results.strip().strip("```").strip("json").strip()
    try:
        json_results = json.loads(results)
    except json.JSONDecodeError as e:
        st.write(f"JSONの解析に失敗しました: {e}")
        return pd.DataFrame()

    if isinstance(json_results, list):
        errors_list = json_results
    else:
        errors_list = json_results.get("errors", [])

    for error in errors_list:
        issue = {
            "ページ番号": page_num,
            "カテゴリ": error.get("category", ""),
            "指摘箇所": error.get("error_location", ""),
            "指摘理由": error.get("reason", ""),
            "周辺テキスト": error.get("context", "")
        }

        if issue["指摘箇所"] or issue["指摘理由"] or issue["周辺テキスト"]:
            issues.append(issue)

    df = pd.DataFrame(issues)

    if not df.empty:
        df.replace("", pd.NA, inplace=True)
        df.dropna(how='all', inplace=True)

    return df

# 全体のテキストから総文数をカウントする関数
def count_total_sentences(page_texts):
    total_text = ''
    for page_num, text in page_texts:
        total_text += text + '\n'
    # 日本語の文末句読点で分割
    sentences = re.split(r'[。．！？\n]', total_text)
    # 空の文字列を除外
    sentences = [s for s in sentences if s.strip()]
    total_sentences = len(sentences)
    return total_sentences

# エラーをGPT-4oを用いて比較する関数
def compare_errors_with_gpt(ai_error, error_list_error):
    prompt = f"""
以下の2つのエラー情報が、同じ誤りを指摘しているかどうかを判断してください。

AIの指摘:
- ページ番号: {ai_error['ページ番号']}
- 指摘箇所: {ai_error['指摘箇所']}
- 指摘理由: {ai_error['指摘理由']}
- 周辺テキスト: {ai_error['周辺テキスト']}

エラーリストの誤記:
- ページ番号: {error_list_error['ページ']}
- 誤記内容: {error_list_error['誤記内容']}
- 正しい内容: {error_list_error['正しい内容']}
- 補足: {error_list_error.get('補足', '')}

これらのエラーは同じ誤りを指摘していますか？「はい」または「いいえ」で答えてください。
"""
    session_message = [{"role": "user", "content": prompt}]

    # チャット履歴に追加
    chat_history.append({"comparison": {"ai_error": ai_error, "error_list_error": error_list_error, "prompt": prompt}})

    for attempt in range(3):  # 最大3回リトライ
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                seed=42,
                temperature=0,
                max_tokens=10,
                messages=session_message
            )
            answer = response.choices[0].message.content.strip()
            # チャット履歴にAIの応答を追加
            chat_history[-1]["comparison"]["response"] = answer
            if "はい" in answer:
                return True
            else:
                return False
        except Exception as e:
            st.write(f"エラーが発生しました（エラー比較中）。再試行します... ({e})")
            time.sleep(5)
    st.write("エラーが続いたため、この比較をスキップします。")
    # チャット履歴にエラー情報を追加
    chat_history[-1]["comparison"]["response"] = f"エラー: {e}"
    return False

# StreamlitでUIを作成
st.title("PDF文章チェックシステム")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")
error_list_file = st.file_uploader("エラー一覧ファイルをアップロードしてください（CSVまたはExcel形式）", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.write("PDFファイルがアップロードされました")

    # PDF全体の処理時間を計測開始
    total_start_time = time.time()

    # PDFからテキストをページごとに抽出
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page_texts = extract_text_with_pymupdf4llm(doc)

    # チャンクに分割
    page_chunks = split_text_into_chunks_by_page(page_texts)

    # デバッグ: チャンクが意図通り分割されているか確認
    display_chunks_debug(page_chunks)

    # 各チャンクごとにAzure OpenAIでテキストチェック
    all_df_results = pd.DataFrame()
    for i, (page_num, chunk_text) in enumerate(page_chunks):
        st.write(f"ページ {page_num} のチャンクをチェックしています...")

        # チャンクが空でないことを確認
        if chunk_text.strip():
            chunk_start_time = time.time()  # チャンクの処理時間計測開始

            check_result = check_text_with_openai(chunk_text, page_num)
            df_results = parse_json_results_to_dataframe(check_result, page_num)
            all_df_results = pd.concat([all_df_results, df_results], ignore_index=True)

            chunk_end_time = time.time()  # チャンクの処理時間計測終了
            st.write(f"ページ {page_num} の処理にかかった時間: {chunk_end_time - chunk_start_time:.2f} 秒")
        else:
            st.write(f"ページ {page_num} のチャンクは空なのでスキップされました。")

        # リクエスト間に待機時間を追加
        time.sleep(1)  # 1秒待機

    # チェック結果を表示（テーブル形式で見やすく）
    st.subheader("AIによるチェック結果")

    if all_df_results.empty:
        st.write("チェック結果が空です")
    else:
        st.dataframe(all_df_results)

    # 全体の処理時間を計測終了
    total_end_time = time.time()
    st.write(f"PDF全体の処理にかかった時間: {total_end_time - total_start_time:.2f} 秒")

    # データをExcelファイルに変換してダウンロード
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        all_df_results.to_excel(writer, index=False)
    processed_data = output.getvalue()

    # ダウンロードオプション
    st.download_button(
        label="結果をExcelでダウンロード",
        data=processed_data,
        file_name="チェック結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # チャット履歴と評価結果をExcelファイルに保存
    chat_output = BytesIO()
    with pd.ExcelWriter(chat_output, engine='xlsxwriter') as writer:
        # チャット履歴をDataFrameに変換
        chat_df_list = []
        for chat in chat_history:
            if 'messages' in chat:
                page_num = chat['page_num']
                user_message = chat['messages'][1]['content']
                assistant_response = chat.get('response', '')
                chat_df_list.append({
                    'ページ番号': page_num,
                    'ユーザーの入力': user_message,
                    'AIの応答': assistant_response
                })
            elif 'comparison' in chat:
                comparison = chat['comparison']
                ai_error = comparison['ai_error']
                error_list_error = comparison['error_list_error']
                prompt = comparison['prompt']
                response = comparison.get('response', '')
                chat_df_list.append({
                    '比較': 'エラー比較',
                    'AIの指摘': json.dumps(ai_error, ensure_ascii=False),
                    'エラーリストの誤記': json.dumps(error_list_error.to_dict(), ensure_ascii=False),
                    'プロンプト': prompt,
                    'AIの応答': response
                })
        chat_df = pd.DataFrame(chat_df_list)
        chat_df.to_excel(writer, sheet_name='チャット履歴', index=False)

        # AIのチェック結果を保存
        all_df_results.to_excel(writer, sheet_name='AIのチェック結果', index=False)
    chat_output.seek(0)
    chat_data = chat_output.getvalue()

    # ダウンロードオプション
    st.download_button(
        label="チャット履歴と評価結果をダウンロード",
        data=chat_data,
        file_name="チャット履歴と評価結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # エラー一覧ファイルがアップロードされている場合、精度評価を行う
    if error_list_file is not None:
        # エラー一覧ファイルを読み込む
        if error_list_file.name.endswith('.csv'):
            try:
                error_df = pd.read_csv(error_list_file, encoding='utf-8')
            except UnicodeDecodeError:
                error_df = pd.read_csv(error_list_file, encoding='shift_jis')
        elif error_list_file.name.endswith('.xlsx'):
            error_df = pd.read_excel(error_list_file)
        else:
            st.write("サポートされていないファイル形式です。CSVまたはExcelファイルをアップロードしてください。")

        # データの前処理：文字列型に変換し、空白を削除
        all_df_results['指摘箇所'] = all_df_results['指摘箇所'].astype(str).str.strip()
        error_df['誤記内容'] = error_df['誤記内容'].astype(str).str.strip()

        # ページ番号を整数型に変換
        all_df_results['ページ番号'] = all_df_results['ページ番号'].astype(int)
        error_df['ページ'] = error_df['ページ'].astype(int)

        # エラーをページごとにグループ化
        ai_errors_by_page = all_df_results.groupby('ページ番号')
        error_list_by_page = error_df.groupby('ページ')

        matched_ai_errors = set()
        matched_error_list_errors = set()

        # 各ページごとにエラーを比較
        for page_num, ai_errors_on_page in ai_errors_by_page:
            if page_num in error_list_by_page.groups:
                error_list_errors_on_page = error_list_by_page.get_group(page_num)
            else:
                error_list_errors_on_page = pd.DataFrame()

            # エラーリストに該当ページのエラーがない場合、FPとしてカウント
            if error_list_errors_on_page.empty:
                continue

            # 各AIの指摘とエラーリストの誤記を比較
            for idx_ai, ai_error in ai_errors_on_page.iterrows():
                for idx_el, error_list_error in error_list_errors_on_page.iterrows():
                    # すでにマッチしている場合はスキップ
                    if idx_el in matched_error_list_errors:
                        continue

                    # GPT-4oを用いて比較
                    if compare_errors_with_gpt(ai_error, error_list_error):
                        matched_ai_errors.add(idx_ai)
                        matched_error_list_errors.add(idx_el)
                        break  # マッチしたら次のAIエラーへ

                    # リクエスト間に待機時間を追加
                    time.sleep(1)  # 1秒待機

        # TPの計算
        TP = len(matched_ai_errors)

        # FPの計算
        FP = len(all_df_results) - TP

        # FNの計算
        FN = len(error_df) - len(matched_error_list_errors)

        # 総文数のカウント
        total_sentences = count_total_sentences(page_texts)

        # 正しい単位の数
        N_correct_units = total_sentences - len(error_df)

        # TNの計算
        TN = N_correct_units - FP

        # 精度評価結果を表示
        st.subheader("精度評価結果")
        st.write(f"TP（適切と思われる件数）: {TP}")
        st.write(f"FP（AIは指摘したが、誤記リストに無い数）: {FP}")
        st.write(f"FN（見逃されたエラーの数）: {FN}")
        st.write(f"TN（正しくエラーなしと判定された正しい部分の数）: {TN}")

        # 精度評価結果をExcelに保存
        evaluation_output = BytesIO()
        with pd.ExcelWriter(evaluation_output, engine='xlsxwriter') as writer:
            # 精度評価結果をDataFrameにまとめる
            evaluation_df = pd.DataFrame({
                '指標': ['TP', 'FP', 'FN', 'TN'],
                '値': [TP, FP, FN, TN]
            })
            evaluation_df.to_excel(writer, sheet_name='精度評価結果', index=False)

            # マッチしたエラーを保存
            matched_ai_df = all_df_results.loc[list(matched_ai_errors)]
            matched_ai_df.to_excel(writer, sheet_name='マッチしたAIエラー', index=False)

            matched_error_list_df = error_df.loc[list(matched_error_list_errors)]
            matched_error_list_df.to_excel(writer, sheet_name='マッチした誤記リスト', index=False)
        evaluation_output.seek(0)
        evaluation_data = evaluation_output.getvalue()

        # ダウンロードオプション
        st.download_button(
            label="精度評価結果をダウンロード",
            data=evaluation_data,
            file_name="精度評価結果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.write("エラー一覧ファイルがアップロードされていません。精度評価を行うには、エラー一覧ファイルをアップロードしてください。")
