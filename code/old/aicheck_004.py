import os
import streamlit as st
import fitz  # PyMuPDF
from openai import AzureOpenAI
import pandas as pd
from io import BytesIO
import json
import tiktoken
import time
import re
import unicodedata

# 環境変数からAPIキーとエンドポイントを取得
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

# チャット履歴を保存するリスト
chat_history = []

# ページごとの処理時間を保存するリスト
page_processing_times = []

# 抽出したテキストを保存するリスト
extracted_texts = []

# チャットログを保存するリスト
chat_logs = []

# JSON部分を抽出する関数
def extract_json(text):
    json_pattern = r'(\{.*\}|\[.*\])'
    matches = re.findall(json_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                json.loads(match)
                return match
            except json.JSONDecodeError:
                continue
    return None

# 応答から制御文字を除去する関数
def sanitize_response(text):
    # 制御文字を除去
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch == '\n' or ch == '\t')
    return text

# チャット形式でOpenAIに問い合わせる関数
def check_text_with_openai(text, page_num):
    session_message = []
    system_prompt = f"""
あなたは建設機械のマニュアルをチェックする専門的なアシスタントです。以下の指示に従って、テキストのチェックを行ってください。
これは技術文書なので、文法については厳密なチェックは不要です。

**チェック項目**:
1. **誤字の確認**: 誤字を指摘してください。
2. **文脈上の誤りの特定**: 記載内容が明確に間違っていて、著しい誤解を招く可能性がある場合を指摘してください。
3. **重大な文法エラーの指摘**: 誤解を生む可能性が高い、重大な文法ミスを指摘してください。

**指摘を行う際のガイドライン**:
- 判断に迷った場合や確信が持てない場合は、偽陽性を避けるため指摘を控えてください。
- **指摘してはいけない項目**:
  - 不自然な改行や空白（PDF抽出時の仕様）
  - 文法的に文章が省略されているが、意味が伝わる場合
  - カタカナ語の長音符に関するもの（独自ルールあり。例：バッテリ、モータ）

**回答形式**:
- 指摘内容は**日本語のJSON形式**で提供してください。
- **JSON以外の文字列は一切含めないでください**。説明や追加情報も不要です。
- 回答では、コードブロックやバッククォート（）は使用しないでください。

**JSONのフォーマット**:

[
  {{
    "page": <ページ番号（整数）>,
    "category": "<カテゴリ>",
    "reason": "<指摘理由>",
    "error_location": "<指摘箇所>",
    "context": "<周辺テキスト>",
    "importance": <重要度（1〜5）>,
    "confidence": <自信度（1〜5）>
  }},
  ...
]

- 各フィールドの説明:
  - "page": 該当ページ番号（整数）
  - "category": 指摘のカテゴリ（例: "誤字", "文法", "文脈"）
  - "reason": 指摘理由（簡潔に、どう修正すべきか）
  - "error_location": 指摘箇所
  - "context": 周辺テキスト
  - "importance": 指摘の重要度（1〜5、5が最も重要）
  - "confidence": 指摘根拠への自信度（1〜5、5が最も自信がある）
"""

    session_message.append({"role": "system", "content": system_prompt})

    prompt = f"""
以下の文章をチェックしてください（該当ページ: {page_num}）。

{text}
"""

    session_message.append({"role": "user", "content": prompt})

    # チャットログに追加
    chat_logs.append({
        "page_num": page_num,
        "messages": session_message.copy()  # メッセージのコピーを保存
    })

    e = None  # eを初期化

    for attempt in range(3):  # 最大3回リトライ
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                seed=42,
                temperature=0,
                max_tokens=2000,
                messages=session_message
            )
            # チャット履歴にAIの応答を追加
            reply_content = response.choices[0].message.content
            # 応答をサニタイズ
            sanitized_reply = sanitize_response(reply_content)
            # JSONを抽出
            json_text = extract_json(sanitized_reply)
            if json_text:
                try:
                    json_results = json.loads(json_text)
                    # JSONのパースに成功した場合
                    chat_history.append({
                        "page_num": page_num,
                        "messages": session_message,
                        "response": sanitized_reply
                    })
                    chat_logs[-1]["response"] = sanitized_reply
                    return sanitized_reply
                except json.JSONDecodeError as json_e:
                    st.write(f"JSONの解析に失敗しました。再試行します... (エラー: {json_e})")
                    if attempt == 0:
                        st.write("デバッグ情報: モデルからの応答が無効なJSON形式でした。応答内容を記録します。")
                        st.write(f"応答内容: {sanitized_reply}")
                    chat_logs[-1]["invalid_response"] = sanitized_reply
                    # モデルに再度指示
                    session_message.append({"role": "user", "content": "先ほどの回答は無効なJSON形式でした。指示に従い、正しいJSON形式のみで回答してください。"})
                    time.sleep(5)
            else:
                st.write("JSONが見つかりませんでした。再試行します...")
                if attempt == 0:
                    st.write("デバッグ情報: モデルからの応答にJSONが見つかりませんでした。応答内容を記録します。")
                    st.write(f"応答内容: {sanitized_reply}")
                chat_logs[-1]["invalid_response"] = sanitized_reply
                session_message.append({"role": "user", "content": "回答にJSONが含まれていませんでした。指示に従い、JSON形式のみで回答してください。"})
                time.sleep(5)
        except Exception as e:
            st.write(f"エラーが発生しました。再試行します... ({e})")
            time.sleep(5)
    st.write("エラーが続いたため、このチャンクをスキップします。")
    # チャット履歴にエラー情報を追加
    if e is not None:
        chat_history.append({
            "page_num": page_num,
            "messages": session_message,
            "response": f"エラー: {e}"
        })
        chat_logs[-1]["response"] = f"エラー: {e}"
    else:
        chat_history.append({
            "page_num": page_num,
            "messages": session_message,
            "response": "不明なエラーが発生しました。"
        })
        chat_logs[-1]["response"] = "不明なエラーが発生しました。"
    return ""

# テキスト前処理関数
def preprocess_text(text):
    # Unicode正規化
    text = unicodedata.normalize('NFKC', text)
    # 不要なスペースを削除（ただし改行は維持）
    text = re.sub(r'[ \t]+', ' ', text)
    # 改行の処理
    text = re.sub(r'([。．！？])\n', r'\1', text)
    text = re.sub(r'(?<![。．！？])\n', '', text)
    text = re.sub(r'([。．！？])', r'\1\n', text)
    # 制御文字を除去
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch == '\n' or ch == '\t')
    return text.strip()

# 矩形の重なりを判定する関数
def rects_overlap(rect1, rect2):
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    if x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1:
        return False
    else:
        return True

# ヘッダーとフッター領域に含まれるかを判定する関数
def is_in_header_footer(block_bbox, page_height, header_height, footer_height):
    x0, y0, x1, y1 = block_bbox
    if y0 <= header_height:
        return True
    if y1 >= page_height - footer_height:
        return True
    return False

# イラスト内のテキストとヘッダー・フッターを除外してテキストを抽出する関数
def extract_text_excluding_images_and_header_footer(pdf_document):
    page_texts = []
    for page_num in range(len(pdf_document)):
        try:
            page_start_time = time.time()
            page = pdf_document[page_num]
            page_height = page.rect.height
            header_height = page_height * 0.05
            footer_height = page_height * 0.05
            blocks = page.get_text("dict")["blocks"]
            image_bboxes = [block["bbox"] for block in blocks if block["type"] == 1]
            text_content = ""
            for block in blocks:
                if block["type"] == 0:
                    block_bbox = block["bbox"]
                    overlaps_image = any(rects_overlap(block_bbox, image_bbox) for image_bbox in image_bboxes)
                    in_header_footer = is_in_header_footer(block_bbox, page_height, header_height, footer_height)
                    if not overlaps_image and not in_header_footer:
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            text_content += line_text + "\n"
                        text_content += "\n"
            text_content = preprocess_text(text_content)
            page_texts.append((page_num + 1, text_content))
            page_end_time = time.time()
            processing_time = page_end_time - page_start_time
            page_processing_times.append({'ページ番号': page_num + 1, '処理時間（秒）': processing_time})
            extracted_texts.append({
                'ページ番号': page_num + 1,
                'テキスト': text_content
            })
        except Exception as e:
            st.write(f"ページ {page_num + 1} の処理中にエラーが発生しました: {e}")
            page_texts.append((page_num + 1, ""))
            page_end_time = time.time()
            processing_time = page_end_time - page_start_time
            page_processing_times.append({'ページ番号': page_num + 1, '処理時間（秒）': processing_time})
            extracted_texts.append({
                'ページ番号': page_num + 1,
                'テキスト': ""
            })
    return page_texts

# テキストをトークン数に基づいてチャンクに分割する関数
def split_text_into_chunks_by_page(page_texts, chunk_size=2000, chunk_overlap=200):
    encoding = tiktoken.encoding_for_model('gpt-4o')
    page_chunks = []
    for page_num, text in page_texts:
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            page_chunks.append((page_num, chunk_text))
    return page_chunks

# チェック結果をパースしてDataFrameに変換する関数
def parse_json_results_to_dataframe(results, page_num):
    issues = []
    results = results.strip()
    json_text = extract_json(results)
    if json_text:
        try:
            json_results = json.loads(json_text)
        except json.JSONDecodeError as e:
            st.write(f"JSONの解析に失敗しました: {e}")
            return pd.DataFrame()
    else:
        st.write("JSONが見つかりませんでした。")
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
            "周辺テキスト": error.get("context", ""),
            "重要度": error.get("importance", ""),
            "自信度": error.get("confidence", "")
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
    sentences = re.split(r'[。．！？\n]', total_text)
    sentences = [s for s in sentences if s.strip()]
    total_sentences = len(sentences)
    return total_sentences

# エラーをGPT-4oを用いて比較する関数
def compare_errors_with_gpt(ai_error, error_list_error):
    prompt = f'''
以下の2つのエラー情報が、同じ誤りを指摘しているかを判断してください。

**判断基準：**

- **エラーの種類が同じであること：** 例）誤字同士、文法エラー同士など
- **指摘箇所が同一または近いこと：** テキスト内での位置が近接している
- **誤記内容に対し指摘理由が適切であること：** 誤記の内容とAIの指摘内容が正確

**注意事項：**

- 判断に迷う場合や確信が持てない場合は「いいえ」と答えてください。
- 回答は「はい」または「いいえ」のみとし、理由や追加の情報は含めないでください。

---

**AIの指摘：**

- **ページ番号：** {ai_error['ページ番号']}
- **カテゴリ：** {ai_error['カテゴリ']}
- **指摘箇所：** {ai_error['指摘箇所']}
- **指摘理由：** {ai_error['指摘理由']}
- **周辺テキスト：** {ai_error['周辺テキスト']}

**誤記リストのエラー：**

- **ページ番号：** {error_list_error['ページ']}
- **誤記内容：** {error_list_error['誤記内容']}
- **正しい内容：** {error_list_error['正しい内容']}
- **補足：** {error_list_error.get('補足', '')}

---

以上の情報に基づいて、これらのエラーは同じ誤りを指摘していますか？「はい」または「いいえ」で答えてください。
'''
    session_message = [{"role": "user", "content": prompt}]

    # チャットログに追加
    chat_logs.append({
        "comparison": {
            "ai_error": ai_error,
            "error_list_error": error_list_error,
            "prompt": prompt
        }
    })

    e = None  # eを初期化

    for attempt in range(3):  # 最大3回リトライ
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                seed=42,
                temperature=0,
                max_tokens=2,
                messages=session_message
            )
            answer = response.choices[0].message.content.strip()
            # 応答の検証
            if answer in ["はい", "いいえ"]:
                chat_history.append({
                    "comparison": {
                        "ai_error": ai_error,
                        "error_list_error": error_list_error,
                        "prompt": prompt,
                        "response": answer
                    }
                })
                chat_logs[-1]["comparison"]["response"] = answer
                return answer == "はい"
            else:
                st.write(f"期待されない回答が返されました: {answer}. 再試行します...")
                if attempt == 0:
                    st.write("デバッグ情報: モデルからの応答が期待した形式ではありませんでした。応答内容を記録します。")
                    st.write(f"応答内容: {answer}")
                chat_logs[-1]["invalid_response"] = answer
                # モデルに再度回答するように指示
                session_message.append({"role": "user", "content": "回答は「はい」または「いいえ」のみで答えてください。理由や追加の情報は含めないでください。"})
                time.sleep(5)
        except Exception as e:
            st.write(f"エラーが発生しました（エラー比較中）。再試行します... ({e})")
            time.sleep(5)
    st.write("エラーが続いたため、この比較をスキップします。")
    if e is not None:
        chat_history.append({
            "comparison": {
                "ai_error": ai_error,
                "error_list_error": error_list_error,
                "prompt": prompt,
                "response": f"エラー: {e}"
            }
        })
        chat_logs[-1]["comparison"]["response"] = f"エラー: {e}"
    else:
        chat_history.append({
            "comparison": {
                "ai_error": ai_error,
                "error_list_error": error_list_error,
                "prompt": prompt,
                "response": "不明なエラーが発生しました。"
            }
        })
        chat_logs[-1]["comparison"]["response"] = "不明なエラーが発生しました。"
    return False

# ダウンロード用のデータをセッション状態で保持するための初期化
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None

if 'evaluation_data' not in st.session_state:
    st.session_state['evaluation_data'] = None

# StreamlitでUIを作成
st.title("PDF文章チェックシステム")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")
error_list_file = st.file_uploader("エラー一覧ファイルをアップロードしてください（CSVまたはExcel形式）", type=["csv", "xlsx"])

if uploaded_file is not None:
    st.write("PDFファイルがアップロードされました")

    # PDF全体の処理時間を計測開始
    total_start_time = time.time()

    # PDFからテキストをページごとに抽出（イラスト内のテキストとヘッダー・フッターを除外）
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page_texts = extract_text_excluding_images_and_header_footer(doc)

    # 読み込んだページ数を表示
    num_pages = len(page_texts)
    st.write(f"読み込んだページ数: {num_pages}")

    # チャンクに分割
    page_chunks = split_text_into_chunks_by_page(page_texts)

    # チャンク分割数を表示
    num_chunks = len(page_chunks)
    st.write(f"チャンク分割数: {num_chunks}")

    # チェック進捗状況の表示
    st.subheader("チェック進捗状況")
    check_progress_bar = st.progress(0)
    check_status_text = st.empty()

    # 各チャンクごとにAzure OpenAIでテキストチェック
    all_df_results = pd.DataFrame()
    for i, (page_num, chunk_text) in enumerate(page_chunks):
        check_status_text.write(f"ページ {page_num} のチャンク {i+1}/{num_chunks} をチェックしています...")

        # チャンクが空でないことを確認
        if chunk_text.strip():
            chunk_start_time = time.time()  # チャンクの処理時間計測開始

            check_result = check_text_with_openai(chunk_text, page_num)
            if check_result:
                df_results = parse_json_results_to_dataframe(check_result, page_num)
                all_df_results = pd.concat([all_df_results, df_results], ignore_index=True)
            else:
                st.write(f"ページ {page_num} のチャンク {i+1} のチェックがスキップされました。")

            chunk_end_time = time.time()  # チャンクの処理時間計測終了
            chunk_processing_time = chunk_end_time - chunk_start_time
        else:
            st.write(f"ページ {page_num} のチャンクは空なのでスキップされました。")

        # プログレスバーの更新
        check_progress = (i + 1) / num_chunks
        check_progress_bar.progress(check_progress)

        # リクエスト間に待機時間を追加
        time.sleep(1)  # 1秒待機

    # チェック進捗状況の完了表示
    check_status_text.write("チェックが完了しました。")

    # 全体の処理時間を計測終了
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    st.write(f"PDF全体の処理にかかった時間: {total_processing_time:.2f} 秒")

    # ページごとの処理時間のDataFrameを作成
    page_times_df = pd.DataFrame(page_processing_times)

    # チェック結果を表示（テーブル形式で見やすく）
    st.subheader("AIによるチェック結果")

    if all_df_results.empty:
        st.write("チェック結果が空です")
    else:
        st.dataframe(all_df_results)

    # データをExcelファイルに変換してセッション状態に保存
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        all_df_results.to_excel(writer, sheet_name='チェック結果', index=False)
        page_times_df.to_excel(writer, sheet_name='ページ処理時間', index=False)
        # 全体の処理時間を記録
        summary_df = pd.DataFrame({'合計処理時間（秒）': [total_processing_time]})
        summary_df.to_excel(writer, sheet_name='合計処理時間', index=False)
        # 抽出したテキストを保存
        extracted_texts_df = pd.DataFrame(extracted_texts)
        extracted_texts_df.to_excel(writer, sheet_name='抽出テキスト', index=False)
        # チャットログを保存
        chat_logs_df = pd.DataFrame(chat_logs)
        chat_logs_df.to_excel(writer, sheet_name='チャットログ', index=False)
    processed_data = output.getvalue()
    st.session_state['processed_data'] = processed_data  # セッション状態に保存

    # ダウンロードオプション
    st.download_button(
        label="結果をExcelでダウンロード",
        data=st.session_state['processed_data'],
        file_name="チェック結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # エラー一覧ファイルがアップロードされている場合、精度評価を行う
    if error_list_file is not None:
        st.write("エラー一覧ファイルがアップロードされました。精度評価を開始します。")

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

        # ページ番号のユニークなリストを取得
        all_pages = set(all_df_results['ページ番号']).union(set(error_df['ページ']))

        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives

        matched_ai_errors = set()
        matched_error_list_errors = set()

        # 精度評価進捗状況の表示
        st.subheader("精度評価進捗状況")
        evaluation_progress_bar = st.progress(0)
        evaluation_status_text = st.empty()
        total_comparisons = sum(len(all_df_results[all_df_results['ページ番号'] == page_num]) * len(error_df[error_df['ページ'] == page_num]) for page_num in all_pages)
        comparisons_done = 0

        # 各ページごとにエラーを比較
        for page_num in all_pages:
            ai_errors_on_page = all_df_results[all_df_results['ページ番号'] == page_num]
            error_list_errors_on_page = error_df[error_df['ページ'] == page_num]

            # AIの指摘がある場合
            for idx_ai, ai_error in ai_errors_on_page.iterrows():
                matched = False
                for idx_el, error_list_error in error_list_errors_on_page.iterrows():
                    if idx_el in matched_error_list_errors:
                        continue
                    # GPT-4oを用いて比較
                    if compare_errors_with_gpt(ai_error, error_list_error):
                        TP += 1
                        matched_ai_errors.add(idx_ai)
                        matched_error_list_errors.add(idx_el)
                        matched = True
                        break  # マッチしたら次のAIエラーへ
                    time.sleep(1)  # 1秒待機

                    # 進捗状況の更新
                    comparisons_done += 1
                    evaluation_progress = comparisons_done / total_comparisons if total_comparisons > 0 else 1
                    evaluation_progress_bar.progress(evaluation_progress)
                    evaluation_status_text.write(f"精度評価中... ({comparisons_done}/{total_comparisons})")

                if not matched:
                    FP += 1  # マッチしなかったAIの指摘はFP

            # 誤記リストのエラーで、AIが指摘しなかったものはFN
            for idx_el, error_list_error in error_list_errors_on_page.iterrows():
                if idx_el not in matched_error_list_errors:
                    FN += 1

        # 精度評価進捗状況の完了表示
        evaluation_status_text.write("精度評価が完了しました。")

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

        # 精度評価のサマリを表示
        st.write("精度評価のサマリ:")
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        st.write(f"適合率（Precision）: {precision:.2f}")
        st.write(f"再現率（Recall）: {recall:.2f}")
        st.write(f"F1スコア: {f1_score:.2f}")

        # 誤記リストにはないが、AIが指摘した項目の一覧（FP）
        false_positives = all_df_results.loc[~all_df_results.index.isin(matched_ai_errors)]
        # 誤記リストにあるが、AIが指摘しなかった項目の一覧（FN）
        false_negatives = error_df.loc[~error_df.index.isin(matched_error_list_errors)]

        # 精度評価結果をExcelに保存してセッション状態に保存
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

            # False Positives（AIが指摘したが誤記リストにない）
            false_positives.to_excel(writer, sheet_name='False Positives', index=False)

            # False Negatives（誤記リストにあるがAIが指摘しなかった）
            false_negatives.to_excel(writer, sheet_name='False Negatives', index=False)

            # ページ処理時間を保存
            page_times_df.to_excel(writer, sheet_name='ページ処理時間', index=False)

            # 合計処理時間を保存
            summary_df.to_excel(writer, sheet_name='合計処理時間', index=False)

            # 抽出したテキストを保存
            extracted_texts_df.to_excel(writer, sheet_name='抽出テキスト', index=False)

            # チャットログを保存
            chat_logs_df.to_excel(writer, sheet_name='チャットログ', index=False)

        evaluation_output.seek(0)
        evaluation_data = evaluation_output.getvalue()
        st.session_state['evaluation_data'] = evaluation_data  # セッション状態に保存

        # ダウンロードオプション
        st.download_button(
            label="精度評価結果をダウンロード",
            data=st.session_state['evaluation_data'],
            file_name="精度評価結果.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.write("エラー一覧ファイルがアップロードされていません。精度評価を行うには、エラー一覧ファイルをアップロードしてください。")

else:
    st.write("PDFファイルをアップロードしてください。")