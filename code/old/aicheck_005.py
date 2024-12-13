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

# チャット形式でOpenAIに問い合わせる関数
def check_text_with_openai(text, page_num):
    session_message = []
    system_prompt = "You are a helpful assistant."
    session_message.append({"role": "system", "content": system_prompt})

    prompt = f"""
あなたには建設機械のマニュアルをチェックしてもらいます（該当ページ: {page_num}）。
以下の項目についてチェックし、指摘してください: 
(1) 誤字、脱字
(2) 文脈的に記載内容が間違っていてると考えられる場合
(3) 文法が間違っていて、読者に意図しない誤解を与える場合

ただし、以下の項目については指摘しないでください:
(1) 不自然な改行や空白については、指摘しない
(2) カタカナ単語の最後の長音符については基本的に省略することが正しいので指摘しない。例：バッテリ、モータ

重要なポイント
・各指摘については、以下の形式で**純粋なJSONのみを返し、コードブロックや追加の説明を一切含めないでください**。
- "page": 該当ページ番号
- "category": 指摘のカテゴリ（例: 誤字、文法、文脈）
- "reason": 指摘した理由（簡潔に記述）
- "error_location": 指摘箇所
- "context": 周辺テキスト
- "importance": 指摘の重要度（1から5の整数で評価、5が最も重要）
- "confidence": 指摘根拠への自信度（1から5の整数で評価、5が最も自信がある）

以下の文章についてチェックを行ってください：

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
            chat_history.append({
                "page_num": page_num,
                "messages": session_message,
                "response": response.choices[0].message.content
            })
            # チャットログにAIの応答を追加
            chat_logs[-1]["response"] = response.choices[0].message.content
            return response.choices[0].message.content
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
        # チャットログにもエラー情報を追加
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
    # 制御文字を除去（改行とタブは除く）
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in ['\n', '\t'])
    # 不要なスペースを削除（ただし改行は維持）
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# 矩形の重なりを判定する関数
def rects_overlap(rect1, rect2):
    # rect: [x0, y0, x1, y1]
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    # 矩形が重なっていない場合
    if x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1:
        return False
    else:
        return True

# ヘッダーとフッター領域に含まれるかを判定する関数
def is_in_header_footer(block_bbox, page_height, header_height, footer_height):
    x0, y0, x1, y1 = block_bbox
    # ヘッダー領域
    if y0 <= header_height:
        return True
    # フッター領域
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
            # ヘッダーとフッターの高さ（ページ高さの15%に調整）
            header_height = page_height * 0.05
            footer_height = page_height * 0.05
            # ページのブロック情報を取得
            blocks = page.get_text("dict")["blocks"]
            # 画像ブロックのbboxリストを作成
            image_bboxes = [block["bbox"] for block in blocks if block["type"] == 1]
            text_content = ""
            for block in blocks:
                if block["type"] == 0:  # テキストブロックの場合
                    block_bbox = block["bbox"]
                    # テキストブロックが画像ブロックと重なっているかをチェック
                    overlaps_image = any(rects_overlap(block_bbox, image_bbox) for image_bbox in image_bboxes)
                    # テキストブロックがヘッダー・フッターに含まれるかをチェック
                    in_header_footer = is_in_header_footer(block_bbox, page_height, header_height, footer_height)
                    if not overlaps_image and not in_header_footer:
                        # 重なっていない場合、テキストを追加
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                line_text += span["text"]
                            text_content += line_text + "\n"
                        text_content += "\n"
            # テキストの前処理を適用
            text_content = preprocess_text(text_content)
            page_texts.append((page_num + 1, text_content))
            page_end_time = time.time()
            processing_time = page_end_time - page_start_time
            page_processing_times.append({'ページ番号': page_num + 1, '処理時間（秒）': processing_time})
            # 抽出したテキストを保存
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
            page_chunks.append((page_num, chunk_text))  # ページ番号とチャンクを対応させる
    
    return page_chunks

# チェック結果をパースしてDataFrameに変換する関数
def parse_json_results_to_dataframe(results, page_num):
    import re
    issues = []
    # 前後の不要な文字を削除
    results = results.strip()
    # コードブロックの除去
    results = re.sub(r'^```(?:json)?', '', results)
    results = re.sub(r'```$', '', results)
    results = results.strip()
    # JSON部分を抽出
    match = re.search(r'(\{.*\}|\[.*\])', results, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = results  # そのまま使用
    try:
        json_results = json.loads(json_str)
    except json.JSONDecodeError as e:
        st.write(f"ページ {page_num} のJSON解析に失敗しました: {e}")
        st.write("返答内容:")
        st.write(results)  # デバッグのために返答内容を表示
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

# チェック結果をフィルタリングする関数を追加
def filter_check_results_with_openai(df_results, chunk_size=5):
    filtered_results_list = []
    excluded_results_list = []
    total_results = len(df_results)
    num_chunks = (total_results // chunk_size) + (1 if total_results % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_results)
        df_chunk = df_results.iloc[start_idx:end_idx]

        # チャンクが空の場合はスキップ
        if df_chunk.empty:
            continue

        session_message = []
        system_prompt = "You are a helpful assistant."
        session_message.append({"role": "system", "content": system_prompt})

        # チャンクのチェック結果をJSON形式で文字列化
        check_results_json = df_chunk.to_dict(orient='records')
        check_results_str = json.dumps(check_results_json, ensure_ascii=False, indent=2)

        prompt = f"""
以下のチェック結果リストがあります。

{check_results_str}

あなたは以下の項目について、チェック結果を選別してください。以下の項目に該当するものは除外してください:

(1) 不自然な改行や空白については、指摘しない
(2) カタカナ単語の末尾の長音符の欠落については独自ルールがあるため指摘しない。例：バッテリ、モータ。ただし単語途中の長音符の欠落は指摘は除外しないでください。

結果は以下の形式で**純粋なJSONのみを返し、コードブロックや追加の説明を一切含めないでください**。

- "filtered_results": フィルタリングされたチェック結果のリスト
- "excluded_results": 除外されたチェック結果のリスト
"""
        session_message.append({"role": "user", "content": prompt})

        # チャットログに追加
        chat_logs.append({
            "filtering_chunk": i + 1,
            "messages": session_message.copy()
        })

        e = None  # eを初期化

        for attempt in range(3):  # 最大3回リトライ
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # モデル名を確認（例: "gpt-4"）
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                # チャット履歴にAIの応答を追加
                chat_history.append({
                    "filtering_chunk": i + 1,
                    "messages": session_message,
                    "response": response.choices[0].message.content
                })
                # チャットログにAIの応答を追加
                chat_logs[-1]["response"] = response.choices[0].message.content

                # レスポンスからJSONを解析
                filtered_chunk, excluded_chunk = parse_filter_response(response.choices[0].message.content)

                # リストに追加
                filtered_results_list.extend(filtered_chunk)
                excluded_results_list.extend(excluded_chunk)

                break  # 成功したらループを抜ける
            except Exception as e:
                st.write(f"フィルタリング中にエラーが発生しました。再試行します... ({e})")
                time.sleep(5)
        else:
            st.write("エラーが続いたため、このフィルタリングチャンクをスキップします。")
            # チャット履歴にエラー情報を追加
            if e is not None:
                chat_history.append({
                    "filtering_chunk": i + 1,
                    "messages": session_message,
                    "response": f"エラー: {e}"
                })
                chat_logs[-1]["response"] = f"エラー: {e}"
            else:
                chat_history.append({
                    "filtering_chunk": i + 1,
                    "messages": session_message,
                    "response": "不明なエラーが発生しました。"
                })
                chat_logs[-1]["response"] = "不明なエラーが発生しました。"

    # 最終的な結果をDataFrameに変換
    filtered_results_df = pd.DataFrame(filtered_results_list)
    excluded_results_df = pd.DataFrame(excluded_results_list)

    return filtered_results_df, excluded_results_df

def parse_filter_response(response_text):
    import re

    # レスポンスからJSON部分を抽出
    response_text = response_text.strip()
    # コードブロックの除去
    response_text = re.sub(r'^```(?:json)?', '', response_text)
    response_text = re.sub(r'```$', '', response_text)
    response_text = response_text.strip()

    # JSON部分を抽出
    match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = response_text  # そのまま使用

    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        st.write(f"フィルタリング結果のJSON解析に失敗しました: {e}")
        st.write("フィルタリング返答内容:")
        st.write(response_text)
        return [], []

    filtered_results = json_data.get('filtered_results', [])
    excluded_results = json_data.get('excluded_results', [])

    return filtered_results, excluded_results


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
    prompt = f'''
以下の2つのエラー情報が、同じ誤りを指摘しているかを判断してください。

**判断基準：**

- **エラーの種類が同じであること：** 例）誤字同士、文法エラー同士など
- **指摘箇所が同一または非常に近いこと：** テキスト内での位置が近接している
- **誤記内容と指摘箇所が一致すること：** 誤記の内容とAIの指摘内容が同じ

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
            # チャット履歴にAIの応答を追加
            chat_history.append({
                "comparison": {
                    "ai_error": ai_error,
                    "error_list_error": error_list_error,
                    "prompt": prompt,
                    "response": answer
                }
            })
            # チャットログにもAIの応答を追加
            chat_logs[-1]["comparison"]["response"] = answer
            if answer == "はい":
                return True
            else:
                return False
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

if 'chat_data' not in st.session_state:
    st.session_state['chat_data'] = None

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
            df_results = parse_json_results_to_dataframe(check_result, page_num)
            all_df_results = pd.concat([all_df_results, df_results], ignore_index=True)

            chunk_end_time = time.time()  # チャンクの処理時間計測終了
            chunk_processing_time = chunk_end_time - chunk_start_time
        else:
            st.write(f"ページ {page_num} のチャンクは空なのでスキップされました。")

        # プログレスバーの更新
        check_progress = (i + 1) / num_chunks
        check_progress_bar.progress(check_progress)

        # リクエスト間に待機時間を追加
        time.sleep(1)  # 1秒待機

    check_status_text.write("チェックが完了しました。")

    # チェック結果のフィルタリングを実行
    st.subheader("チェック結果の再検証を実行しています...")

    # フィルタリング処理を実行し、結果を取得
    filtered_results_df, excluded_results_df = filter_check_results_with_openai(all_df_results)

    # フィルタリング後のチェック結果を表示
    if filtered_results_df.empty:
        st.write("フィルタリング後のチェック結果が空です")
    else:
        st.dataframe(filtered_results_df)

    # 全体の処理時間を計測終了
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    st.write(f"PDF全体の処理にかかった時間: {total_processing_time:.2f} 秒")

    # ページごとの処理時間の DataFrame を作成
    page_times_df = pd.DataFrame(page_processing_times)

    # データを Excel ファイルに変換してセッション状態に保存
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        filtered_results_df.to_excel(writer, sheet_name='チェック結果', index=False)
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
        # 除外された結果を保存
        if not excluded_results_df.empty:
            excluded_results_df.to_excel(writer, sheet_name='除外された指摘項目', index=False)

    # ここから `with` ブロックの外側になります
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

        # 除外された結果を精度評価から除外
        if not excluded_results_df.empty:
            filtered_results_df = filtered_results_df[~filtered_results_df.index.isin(excluded_results_df.index)]

        # データの前処理：文字列型に変換し、空白を削除
        filtered_results_df['指摘箇所'] = filtered_results_df['指摘箇所'].astype(str).str.strip()
        error_df['誤記内容'] = error_df['誤記内容'].astype(str).str.strip()

        # ページ番号を整数型に変換
        filtered_results_df['ページ番号'] = filtered_results_df['ページ番号'].astype(int)
        error_df['ページ'] = error_df['ページ'].astype(int)

        # ページ番号のユニークなリストを取得
        all_pages = set(filtered_results_df['ページ番号']).union(set(error_df['ページ']))

        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives

        matched_ai_errors = set()
        matched_error_list_errors = set()

        # 精度評価進捗状況の表示
        st.subheader("精度評価進捗状況")
        evaluation_progress_bar = st.progress(0)
        evaluation_status_text = st.empty()
        total_comparisons = sum(
            len(filtered_results_df[filtered_results_df['ページ番号'] == page_num]) * 
            len(error_df[error_df['ページ'] == page_num]) 
            for page_num in all_pages
        )
        comparisons_done = 0

        # 各ページごとにエラーを比較
        for page_num in all_pages:
            ai_errors_on_page = filtered_results_df[filtered_results_df['ページ番号'] == page_num]
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
        false_positives = filtered_results_df.loc[~filtered_results_df.index.isin(matched_ai_errors)]
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
            matched_ai_df = filtered_results_df.loc[list(matched_ai_errors)]
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

            # 除外された結果を保存
            if not excluded_results_df.empty:
                excluded_results_df.to_excel(writer, sheet_name='除外された指摘項目', index=False)

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

# アップロードされたPDFファイルがない場合のメッセージ
else:
    st.write("PDFファイルをアップロードしてください。")
