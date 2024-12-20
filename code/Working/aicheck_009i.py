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
import asyncio
import base64

st.set_page_config(layout="wide")

key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not key or not endpoint:
    st.error("Azure OpenAIのAPIキーまたはエンドポイントが設定されていません。")

client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

# セッションステート初期化
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False
if "uploaded_pdf_data" not in st.session_state:
    st.session_state.uploaded_pdf_data = None
if "extracted_texts" not in st.session_state:
    st.session_state.extracted_texts = []
if "page_processing_times" not in st.session_state:
    st.session_state.page_processing_times = []
if "chat_logs" not in st.session_state:
    st.session_state.chat_logs = []
if "all_df_results" not in st.session_state:
    st.session_state.all_df_results = pd.DataFrame()
if "filtered_results_df" not in st.session_state:
    st.session_state.filtered_results_df = pd.DataFrame()
if "excluded_results_df" not in st.session_state:
    st.session_state.excluded_results_df = pd.DataFrame()
if "extract_time" not in st.session_state:
    st.session_state.extract_time = 0.0
if "check_time" not in st.session_state:
    st.session_state.check_time = 0.0
if "filtering_time" not in st.session_state:
    st.session_state.filtering_time = 0.0
if "total_processing_time" not in st.session_state:
    st.session_state.total_processing_time = 0.0

def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in ['\n', '\t'])
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'([ぁ-んァ-ン一-龥])\s+([ぁ-んァ-ン一-龥])', r'\1\2', text)
    return text.strip()

def rects_overlap(rect1, rect2):
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    if x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1:
        return False
    return True

def is_in_header_footer(block_bbox, page_height, header_height, footer_height):
    _, y0, _, y1 = block_bbox
    if y0 <= header_height:
        return True
    if y1 >= page_height - footer_height:
        return True
    return False

def extract_text_excluding_images_and_header_footer(pdf_document):
    page_texts = []
    total_pages = len(pdf_document)

    extract_progress_bar = st.progress(0)
    extract_status_text = st.empty()
    extract_start_time = time.time()

    page_processing_times_local = []
    extracted_texts_local = []

    for page_num in range(total_pages):
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
                    overlaps_image = any(rects_overlap(block_bbox, ib) for ib in image_bboxes)
                    in_hf = is_in_header_footer(block_bbox, page_height, header_height, footer_height)
                    if not overlaps_image and not in_hf:
                        for line in block["lines"]:
                            line_text = "".join(span["text"] for span in line["spans"])
                            text_content += line_text + "\n"
                        text_content += "\n"
            text_content = preprocess_text(text_content)
            actual_page_num = page_num + 1
            page_texts.append((actual_page_num, text_content))
            processing_time = time.time() - page_start_time
            page_processing_times_local.append({'ページ番号': actual_page_num, '処理時間（秒）': processing_time})
            extracted_texts_local.append({'ページ番号': actual_page_num, 'テキスト': text_content})

            extract_progress = actual_page_num / total_pages
            extract_status_text.text(f"テキスト抽出中... {actual_page_num}/{total_pages}")
            extract_progress_bar.progress(extract_progress)

        except Exception as e:
            st.write(f"ページ {page_num + 1} の処理中にエラーが発生しました: {e}")
            page_texts.append((page_num + 1, ""))
            page_processing_times_local.append({'ページ番号': page_num + 1, '処理時間（秒）': 0})
            extracted_texts_local.append({'ページ番号': page_num + 1, 'テキスト': ""})

    extract_end_time = time.time()
    extract_time_local = extract_end_time - extract_start_time
    extract_status_text.text("テキスト抽出完了")
    st.write(f"テキスト抽出処理時間: {extract_time_local:.2f} 秒")
    return page_texts, extract_time_local, page_processing_times_local, extracted_texts_local

def extract_images_from_pdf(pdf_document):
    """PDFの各ページから画像を抽出し、base64エンコードした画像データを返す"""
    page_images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images_on_page = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]  # 画像の拡張子 (png, jpeg など)
            
            # 画像をbase64にエンコード
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:image/{image_ext};base64,{encoded_image}"
            images_on_page.append(image_data_url)
        
        page_images.append({"page_num": page_num + 1, "images": images_on_page})
    return page_images

def split_text_into_chunks_by_page(page_texts, chunk_size=2000, chunk_overlap=200):
    try:
        encoding = tiktoken.encoding_for_model('gpt-4o')
    except:
        encoding = tiktoken.get_encoding('cl100k_base')
    page_chunks = []
    for page_num, text in page_texts:
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            page_chunks.append((page_num, chunk_text))
    return page_chunks

async def async_check_text_and_images_with_openai(client, text, page_num, images, semaphore):
    # 画像をHTMLイメージタグとして組み込む
    image_tags = ""
    for image_data_url in images:
        image_tags += f'<img src="{image_data_url}" alt="page-{page_num}-image" style="max-width:100%;">\n'

    session_message = [
        {"role": "system", "content": """あなたは建設機械マニュアルのチェックアシスタントです。以下の事項を確認し、問題点を指摘してください。

【指摘してほしい問題点】
(1) 誤字や脱字
(2) 意味伝達に支障があるレベルの文法ミス
(3) 文脈的に不適切、または誤った記載
(4) 画像内の情報に関する問題点（例: 画像内のテキストラベルミス、文書と画像情報の不整合など）

【指摘不要】
- 不自然な改行・空白（PDF抽出由来の可能性が高い）
- カタカナ用語の末尾長音の省略に関する指摘（例：バッテリ→バッテリー）

【回答形式】  
以下の形式で問題がある場合は列挙してください。  
問題がなければ空の配列"[]"のみ出力してください。

[
  {
    "page": <page_num>,
    "category": "誤字" or "文法" or "文脈" or "画像",
    "reason": "具体的な理由",
    "error_location": "指摘箇所",
    "suggestion": "修正案"
  }
]
""" },
        {"role": "user", "content": f"以下は、PDF抽出テキストの一部と画像です（該当ページ: {page_num}）。\n\n【テキスト】:\n{text}\n\n【画像】:\n{image_tags}"}
    ]

    async with semaphore:
        await asyncio.sleep(0.5)  # レート制限対策
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                st.session_state.chat_logs.append({
                    "page_num": page_num,
                    "messages": session_message,
                    "response": response.choices[0].message.content
                })
                return (page_num, response.choices[0].message.content)
            except Exception as ex:
                if attempt < 2:
                    st.write(f"ページ {page_num} のチェック中にエラーが発生しました。再試行します... ({ex})")
                    await asyncio.sleep(5)
                else:
                    st.warning(f"ページ {page_num} のチェックに失敗しました。")
                    st.session_state.chat_logs.append({
                        "page_num": page_num,
                        "messages": session_message,
                        "response": f"エラー: {ex}"
                    })
                    return (page_num, "")

async def run_async_check_with_images(client, page_texts, page_images):
    semaphore = asyncio.Semaphore(5)
    tasks = []
    # ページごとに、テキストと対応する画像リストを取得
    for page_num, text in page_texts:
        images = next((item["images"] for item in page_images if item["page_num"] == page_num), [])
        tasks.append(async_check_text_and_images_with_openai(client, text, page_num, images, semaphore))
    results = await asyncio.gather(*tasks)
    return results

def parse_json_results_to_dataframe(results, page_num):
    if not results.strip():
        return pd.DataFrame()

    try:
        json_results = json.loads(results.strip())
    except json.JSONDecodeError:
        st.write(f"ページ {page_num} のJSON解析に失敗しました。")
        return pd.DataFrame()

    if not isinstance(json_results, list):
        json_results = [json_results]

    issues = []
    for error in json_results:
        issue = {
            "ページ番号": page_num,  
            "カテゴリ": error.get("category", ""),
            "指摘箇所": error.get("error_location", ""),
            "指摘理由": error.get("reason", ""),
            "修正案": error.get("suggestion", "")
        }
        if any(val for val in issue.values()):
            issues.append(issue)

    df = pd.DataFrame(issues)
    if not df.empty:
        df.replace("", pd.NA, inplace=True)
        df.dropna(how='all', inplace=True)
    return df

def parse_filter_response(response_text):
    response_text = response_text.strip()
    response_text = re.sub(r'^```(?:json)?', '', response_text)
    response_text = re.sub(r'```$', '', response_text)
    response_text = response_text.strip()

    match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
    if match:
        json_str = match.group(0)
    else:
        json_str = response_text
    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError:
        return [], []
    filtered_results = json_data.get('filtered_results', [])
    excluded_results = json_data.get('excluded_results', [])
    return filtered_results, excluded_results

async def async_filter_chunk_with_openai(client, df_chunk, chunk_index, semaphore):
    system_prompt = """あなたはチェック結果を整理するフィルタリングアシスタントです。
以下の方針で、チェック結果をふるい分けてください。

【excluded_results に回す項目】  
- 不自然な改行や空白のみの問題
- カタカナ末尾長音有無の揺れのみ

【filtered_results に残す項目】  
- 上記以外の重要な誤字・文法・文脈・画像関連ミス

必ず以下形式のJSONのみを出力してください:
{
  "filtered_results": [...],
  "excluded_results": [...]
}
"""

    check_results_json = df_chunk.to_dict(orient='records')
    check_results_str = json.dumps(check_results_json, ensure_ascii=False, indent=2)

    user_prompt = f"""
以下はチェック結果のリストです。

{check_results_str}

上記ルールに基づき、"filtered_results"と"excluded_results"に振り分けてください。
"""

    session_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    async with semaphore:
        await asyncio.sleep(0.5)
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                st.session_state.chat_logs.append({
                    "chunk_index": chunk_index,
                    "messages": session_message,
                    "response": response.choices[0].message.content
                })
                filtered_chunk, excluded_chunk = parse_filter_response(response.choices[0].message.content)
                return filtered_chunk, excluded_chunk
            except Exception as ex:
                if attempt < 2:
                    st.write(f"フィルタリングチャンク {chunk_index} の処理中にエラーが発生しました。再試行します... ({ex})")
                    await asyncio.sleep(5)
                else:
                    st.warning(f"フィルタリングチャンク {chunk_index} の処理に失敗しました。")
                    st.session_state.chat_logs.append({
                        "chunk_index": chunk_index,
                        "messages": session_message,
                        "response": f"エラー: {ex}"
                    })
                    return [], []

async def run_async_filter(client, df_results, chunk_size, progress_bar):
    filtered_results_list = []
    excluded_results_list = []
    total_results = len(df_results)
    if total_results == 0:
        if progress_bar:
            progress_bar.progress(1.0)
        return pd.DataFrame(), pd.DataFrame(), 0

    filtering_start_time = time.time()
    num_chunks = (total_results // chunk_size) + (1 if total_results % chunk_size != 0 else 0)

    filter_status_text = st.empty()
    filter_status_text.text("フィルタリング中...")
    progress_bar.progress(0.0)

    semaphore = asyncio.Semaphore(5)
    tasks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_results)
        df_chunk = df_results.iloc[start_idx:end_idx]
        if df_chunk.empty:
            continue
        tasks.append((i+1, df_chunk))

    completed = 0
    async_tasks = [async_filter_chunk_with_openai(client, chunk_df, idx, semaphore) for idx, chunk_df in tasks]

    for coro, (idx, _) in zip(asyncio.as_completed(async_tasks), tasks):
        f_chunk, e_chunk = await coro
        filtered_results_list.extend(f_chunk)
        excluded_results_list.extend(e_chunk)
        completed += 1
        progress = completed / len(tasks)
        progress_bar.progress(progress)
        filter_status_text.text(f"フィルタリング中... {completed}/{len(tasks)} チャンク完了")

    filtering_end_time = time.time()
    filtering_time_local = filtering_end_time - filtering_start_time
    filter_status_text.text(f"フィルタリング完了 (処理時間: {filtering_time_local:.2f} 秒)")

    filtered_results_df = pd.DataFrame(filtered_results_list)
    excluded_results_df = pd.DataFrame(excluded_results_list)

    return filtered_results_df, excluded_results_df, filtering_time_local

def add_location_info(df, extracted_texts):
    page_map = {x["ページ番号"]: x["テキスト"] for x in extracted_texts}

    df["行番号"] = None
    df["文字オフセット"] = None

    for i, row in df.iterrows():
        page_num = row.get("ページ番号", None)
        if page_num is None:
            continue
        err_loc = row["指摘箇所"]
        if pd.isna(err_loc) or page_num not in page_map:
            continue

        err_loc = unicodedata.normalize('NFKC', str(err_loc)).strip()
        full_text = unicodedata.normalize('NFKC', page_map[page_num]).strip()

        idx = full_text.find(err_loc)

        if idx == -1:
            words = err_loc.split()
            candidates = [full_text.find(w) for w in words if w and full_text.find(w) != -1]
            if candidates:
                idx = min(candidates)

        if idx == -1:
            continue

        line_number = full_text.count('\n', 0, idx) + 1
        df.at[i, "行番号"] = line_number
        df.at[i, "文字オフセット"] = f"{idx}文字目付近"

    return df

def display_dataframe(df):
    st.dataframe(df)

def create_readable_chat_dataframe(chat_logs_local):
    records = []
    for log in chat_logs_local:
        page_or_chunk = log.get("page_num", log.get("chunk_index", None))
        messages = log.get("messages", [])
        response = log.get("response", "")
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            records.append({
                "ページorチャンク": page_or_chunk,
                "ロール": role,
                "メッセージ内容": content
            })
        if response:
            records.append({
                "ページorチャンク": page_or_chunk,
                "ロール": "assistant",
                "メッセージ内容": response
            })
    return pd.DataFrame(records)

st.title("文章AIチェックシステム_画像対応版")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if uploaded_file is not None and st.session_state.uploaded_pdf_data is None:
    st.session_state.uploaded_pdf_data = uploaded_file.read()

if st.session_state.uploaded_pdf_data is not None and not st.session_state.processing_done:
    total_start_time = time.time()

    doc = fitz.open(stream=st.session_state.uploaded_pdf_data, filetype="pdf")
    page_texts, extract_time_local, page_processing_times_local, extracted_texts_local = extract_text_excluding_images_and_header_footer(doc)
    total_pages = len(doc)
    st.write(f"読み込んだページ数: {len(page_texts)} / {total_pages}")

    st.session_state.page_processing_times = page_processing_times_local
    st.session_state.extracted_texts = extracted_texts_local
    st.session_state.extract_time = extract_time_local

    # 画像抽出
    page_images = extract_images_from_pdf(doc)

    # 非同期でテキスト＋画像チェック
    st.subheader("チェック進捗状況")
    progress_bar = st.progress(0)
    check_status_text = st.empty()

    results = asyncio.run(run_async_check_with_images(client, page_texts, page_images))
    # チェック結果解析
    all_df_results_local = pd.DataFrame()
    for (page_num, result_str) in results:
        df_part = parse_json_results_to_dataframe(result_str, page_num)
        all_df_results_local = pd.concat([all_df_results_local, df_part], ignore_index=True)

    # 重複削除
    if not all_df_results_local.empty:
        duplicate_key_columns = ["ページ番号", "指摘箇所", "指摘理由", "修正案"]
        all_df_results_local.drop_duplicates(subset=duplicate_key_columns, keep='first', inplace=True)

    # 行番号・オフセット付与
    if not all_df_results_local.empty and "ページ番号" in all_df_results_local.columns:
        all_df_results_local = add_location_info(all_df_results_local, st.session_state.extracted_texts)
        all_df_results_local.sort_values(by=["ページ番号"], inplace=True)

        st.write("フィルタリング前のチェック結果:")
        st.dataframe(all_df_results_local)
    else:
        st.write("指摘事項がありませんでした。")
        all_df_results_local = pd.DataFrame()

    st.session_state.all_df_results = all_df_results_local

    # フィルタリング
    if not all_df_results_local.empty:
        st.subheader("チェック結果フィルタリング中...")
        filtering_progress = st.progress(0)
        filtered_results_df_local, excluded_results_df_local, filtering_time_local = asyncio.run(run_async_filter(client, all_df_results_local, chunk_size=5, progress_bar=filtering_progress))
    else:
        filtered_results_df_local = pd.DataFrame()
        excluded_results_df_local = pd.DataFrame()
        filtering_time_local = 0.0

    if not filtered_results_df_local.empty:
        filtered_results_df_local.sort_values(by=["ページ番号"], inplace=True)
    if not excluded_results_df_local.empty:
        excluded_results_df_local.sort_values(by=["ページ番号"], inplace=True)

    st.session_state.filtered_results_df = filtered_results_df_local
    st.session_state.excluded_results_df = excluded_results_df_local
    st.session_state.filtering_time = filtering_time_local

    total_end_time = time.time()
    total_processing_time_local = total_end_time - total_start_time
    st.session_state.total_processing_time = total_processing_time_local

    # 結果表示
    st.write(f"総処理時間: {total_processing_time_local:.2f} 秒")
    st.write(f"テキスト抽出処理時間: {extract_time_local:.2f} 秒")
    st.write(f"チェック処理時間: {(st.session_state.check_time or 0.0):.2f} 秒")  # check_timeは今回はrun_async_check_with_images内で計測したければ同様に計測可能
    st.write(f"フィルタリング処理時間: {filtering_time_local:.2f} 秒")

    if filtered_results_df_local.empty:
        st.write("フィルタリング後に残った指摘事項はありません。")
    else:
        st.write("フィルタリング後のチェック結果:")
        display_dataframe(filtered_results_df_local)

    if not excluded_results_df_local.empty:
        st.write("除外された指摘事項:")
        st.dataframe(excluded_results_df_local)

    st.session_state.processing_done = True

if st.session_state.processing_done:
    excel_filtered = st.session_state.filtered_results_df if not st.session_state.filtered_results_df.empty else pd.DataFrame()
    excel_excluded = st.session_state.excluded_results_df if not st.session_state.excluded_results_df.empty else pd.DataFrame()
    excel_all = st.session_state.all_df_results if not st.session_state.all_df_results.empty else pd.DataFrame()
    excel_chat_logs = pd.DataFrame(st.session_state.chat_logs)
    excel_chat_conversations = create_readable_chat_dataframe(st.session_state.chat_logs)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not excel_filtered.empty:
            excel_filtered.to_excel(writer, sheet_name='フィルタ後チェック結果', index=False)
        if not excel_excluded.empty:
            excel_excluded.to_excel(writer, sheet_name='除外された指摘項目', index=False)
        if not excel_chat_logs.empty:
            excel_chat_logs.to_excel(writer, sheet_name='チャットログ(Raw)', index=False)
        if not excel_chat_conversations.empty:
            excel_chat_conversations.to_excel(writer, sheet_name='チャット内容整形', index=False)
        page_times_df = pd.DataFrame(st.session_state.page_processing_times)
        page_times_df.to_excel(writer, sheet_name='ページ処理時間', index=False)
        summary_df = pd.DataFrame({
            '合計処理時間（秒）': [st.session_state.total_processing_time],
            'テキスト抽出時間（秒）': [st.session_state.extract_time],
            'チェック処理時間（秒）': [st.session_state.check_time],
            'フィルタリング処理時間（秒）': [st.session_state.filtering_time]
        })
        summary_df.to_excel(writer, sheet_name='処理時間サマリ', index=False)
        extracted_texts_df = pd.DataFrame(st.session_state.extracted_texts)
        extracted_texts_df.to_excel(writer, sheet_name='抽出テキスト', index=False)
        if not excel_all.empty:
            excel_all.to_excel(writer, sheet_name='フィルタ前チェック結果', index=False)
    output.seek(0)

    st.download_button(
        label="結果をExcelでダウンロード",
        data=output.getvalue(),
        file_name="チェック結果_フィルタリングあり.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
