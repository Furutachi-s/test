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
import html
import streamlit.components.v1 as components

# ===== 環境変数からAPIキーとエンドポイントを取得 =====
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not key or not endpoint:
    st.error("Azure OpenAIのAPIキーまたはエンドポイントが設定されていません。")

# ===== Azure OpenAI クライアント初期化 =====
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

page_processing_times = []
extracted_texts = []

def preprocess_text(text):
    """テキストの前処理を行います。"""
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in ['\n', '\t'])
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def rects_overlap(rect1, rect2):
    """2つの矩形が重なっているかを判定します。"""
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    if x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2:
        return False
    return True

def is_in_header_footer(block_bbox, page_height, header_height, footer_height):
    """ブロックがヘッダーまたはフッターに含まれているかを判定します。"""
    _, y0, _, y1 = block_bbox
    if y0 <= header_height:
        return True
    if y1 >= page_height - footer_height:
        return True
    return False

def extract_text_excluding_images_and_header_footer(pdf_document):
    """画像とヘッダー・フッターを除外してテキストを抽出します。"""
    page_texts = []
    total_pages = len(pdf_document)

    st.write(f"PDFの総ページ数: {total_pages}")
    extract_progress_bar = st.progress(0)
    extract_status_text = st.empty()
    extract_start_time = time.time()

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
            actual_page_num = page_num + 1  # 1ベースのページ番号
            page_texts.append((actual_page_num, text_content))
            processing_time = time.time() - page_start_time
            page_processing_times.append({'ページ番号': actual_page_num, '処理時間（秒）': processing_time})
            extracted_texts.append({'ページ番号': actual_page_num, 'テキスト': text_content})

            extract_progress = actual_page_num / total_pages
            extract_status_text.text(f"テキスト抽出中... {actual_page_num}/{total_pages}")
            extract_progress_bar.progress(extract_progress)

        except Exception as e:
            st.write(f"ページ {page_num + 1} の処理中にエラーが発生しました: {e}")
            page_texts.append((page_num + 1, ""))
            page_processing_times.append({'ページ番号': page_num + 1, '処理時間（秒）': 0})
            extracted_texts.append({'ページ番号': page_num + 1, 'テキスト': ""})

    extract_end_time = time.time()
    extract_time = extract_end_time - extract_start_time
    extract_status_text.text("テキスト抽出完了")
    st.write(f"テキスト抽出処理時間: {extract_time:.2f} 秒")
    return page_texts, extract_time

def split_text_into_chunks_by_page(page_texts, chunk_size=2000, chunk_overlap=200):
    """ページごとのテキストをチャンクに分割します。"""
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

async def async_check_text_with_openai(client, text, page_num, semaphore):
    """非同期でOpenAI APIを呼び出してテキストをチェックします。"""
    # プロンプト調整：  
    # 必須修正対象がなければ無理に指摘をしないが、必要であれば参考情報を1～2件程度出しても良い。
    # ただしほぼ問題がなければ空配列[]で返しても良い。
    session_message = [
        {"role": "system", "content": """あなたは建設機械マニュアルのチェックアシスタントです。
以下のルールに従って、テキスト中の問題を指摘してください：

【指摘してほしい問題点(必須修正)】
(1) 明確な誤字・脱字
(2) 明確な文法ミス
(3) 文脈的な明確な誤り
→ "info_type": "必須修正"

【指摘不要】
- 不自然な改行・空白
- カタカナ末尾長音省略
- 画像・図版関連指摘

【参考情報 (info_type=参考情報)】
- 同一用語の表記揺れ統一提案
- 改善余地あるが誤りではない表現提案

【追加条件】
- 必須修正対象が一切ない場合、無理に指摘は不要。参考情報も特に思い当たらなければ空の配列"[]"のみ返すこと。
- 必須修正対象が無くても、もし有意な改善提案（表記揺れ統一など）が実際に存在するなら、参考情報として1～2件程度まで指摘しても良い。
- 出力は必ずJSON形式のみ(不要な説明文は出さない)。

【出力形式】
[
  {
    "page": <page_num>,
    "category": "誤字"/"文法"/"文脈"/"参考情報",
    "reason": "指摘理由",
    "error_location": "指摘箇所",
    "context": "周辺テキスト",
    "suggestion": "修正案",
    "importance": 1～5,
    "confidence": 1～5,
    "info_type": "必須修正"または"参考情報"
  }
]
""" },
        {"role": "user", "content": f"""
以下は、PDF抽出テキストの一部です（該当ページ: {page_num}）。

{text}

このページのテキストをチェックし、上記ルールに基づき指摘してください。
出力はJSON形式のみ。
"""}
    ]

    async with semaphore:
        await asyncio.sleep(0.5)  # レート制限対策
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    seed=42,
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                return (page_num, response.choices[0].message.content)
            except Exception as ex:
                await asyncio.sleep(5)
        st.warning(f"ページ {page_num} のチェックに失敗しました。")
        return (page_num, "")

async def run_async_check(client, page_chunks, progress_bar, check_status_text):
    """非同期でテキストチェックを実行します。"""
    non_empty_chunks = [(p, t) for p, t in page_chunks if t.strip()]
    semaphore = asyncio.Semaphore(5)
    
    tasks = [async_check_text_with_openai(client, chunk_text, page_num, semaphore)
             for (page_num, chunk_text) in non_empty_chunks]
    
    tasks = [asyncio.create_task(task) for task in tasks]
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        check_status_text.write("チェック対象がありません。")
        return [], non_empty_chunks, 0

    completed = 0
    check_start_time = time.time()
    results = []
    check_status_text.text("チェック処理を開始します...")
    progress_bar.progress(0.0)

    for task in asyncio.as_completed(tasks):
        page_num, result = await task
        results.append((page_num, result))
        completed += 1
        progress = completed / total_tasks
        progress_bar.progress(progress)
        check_status_text.text(f"チェック中... {completed}/{total_tasks} チャンク完了")

    check_end_time = time.time()
    check_time = check_end_time - check_start_time
    check_status_text.text(f"チェックが完了しました！ (処理時間: {check_time:.2f} 秒)")
    progress_bar.progress(1.0)

    return results, non_empty_chunks, check_time

def parse_json_results_to_dataframe(results, page_num):
    """JSON形式の結果をDataFrameに変換します。"""
    if not results.strip():
        return pd.DataFrame()

    try:
        json_results = json.loads(results.strip())
    except json.JSONDecodeError:
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
            "修正案": error.get("suggestion", ""),
            "周辺テキスト": error.get("context", ""),
            "重要度": error.get("importance", ""),
            "自信度": error.get("confidence", ""),
            "情報種別": error.get("info_type", "")
        }
        if any(val for val in issue.values()):
            issues.append(issue)

    df = pd.DataFrame(issues)
    if not df.empty:
        df.replace("", pd.NA, inplace=True)
        df.dropna(how='all', inplace=True)
    return df

def parse_filter_response(response_text):
    """フィルタリング用のレスポンスを解析します。"""
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
    """非同期でフィルタリングを実行します。"""
    session_message = []
    system_prompt = "You are a helpful assistant."
    session_message.append({"role": "system", "content": system_prompt})

    check_results_json = df_chunk.to_dict(orient='records')
    check_results_str = json.dumps(check_results_json, ensure_ascii=False, indent=2)

    prompt = f"""
以下のチェック結果リストがあります。

{check_results_str}

以下は除外：
- 不自然な改行・空白
- カタカナ末尾長音省略指摘
- 画像・図版関連指摘

JSONで:
"filtered_results": 除外でない結果
"excluded_results": 除外結果
"""
    session_message.append({"role": "user", "content": prompt})

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
                filtered_chunk, excluded_chunk = parse_filter_response(response.choices[0].message.content)
                return filtered_chunk, excluded_chunk
            except Exception as ex:
                await asyncio.sleep(5)
        st.warning(f"フィルタリングチャンク {chunk_index} の処理に失敗しました。")
        return [], []

async def run_async_filter(client, df_results, chunk_size, progress_bar):
    """非同期でフィルタリングを実行します。"""
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
    filtering_time = filtering_end_time - filtering_start_time
    filter_status_text.text(f"フィルタリング完了 (処理時間: {filtering_time:.2f} 秒)")

    filtered_results_df = pd.DataFrame(filtered_results_list)
    excluded_results_df = pd.DataFrame(excluded_results_list)

    return filtered_results_df, excluded_results_df, filtering_time

def add_location_info(df, extracted_texts):
    """指摘箇所の行番号と文字オフセット情報を追加します。"""
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

        # 正規化と前後空白除去
        err_loc = unicodedata.normalize('NFKC', str(err_loc)).strip()
        full_text = unicodedata.normalize('NFKC', page_map[page_num]).strip()

        idx = full_text.find(err_loc)

        if idx == -1:
            # 近似値検索: 単語で探す
            words = err_loc.split()
            candidates = [full_text.find(w) for w in words if w and full_text.find(w) != -1]
            if candidates:
                idx = min(candidates)

        if idx == -1:
            # 見つからない場合はスキップ
            continue

        line_number = full_text.count('\n', 0, idx) + 1
        df.at[i, "行番号"] = line_number
        df.at[i, "文字オフセット"] = f"{idx}文字目付近"

    return df

def copy_button(text, button_label="コピー", key=None):
    """テキストをコピーするボタンを作成します。"""
    escaped_text = html.escape(text.replace("'", "\\'"))
    button_id = f"copy-button-{key}" if key else "copy-button"
    html_code = f"""
    <button id="{button_id}" onclick="navigator.clipboard.writeText('{escaped_text}')">{button_label}</button>
    """
    components.html(html_code, height=30, scrolling=False)

def display_dataframe_with_copy(df):
    """DataFrameを表示し、各行にコピー機能を追加します。"""
    display_df = df.drop(columns=["重要度", "自信度", "情報種別"], errors='ignore')

    for index, row in display_df.iterrows():
        cols = st.columns(len(display_df.columns) + 1)  # +1 for the copy button
        for i, col in enumerate(display_df.columns):
            cols[i].write(row[col] if pd.notna(row[col]) else "")
        copy_text = row["指摘箇所"] if "指摘箇所" in row else ""
        if pd.notna(copy_text):
            with cols[-1]:
                copy_button(str(copy_text), key=f"copy_{index}")
        else:
            cols[-1].write("")

st.title("文章AIチェックシステム_調整版")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if uploaded_file is not None:
    st.write("PDFファイルがアップロードされました")
    total_start_time = time.time()

    # PDFを開く
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page_texts, extract_time = extract_text_excluding_images_and_header_footer(doc)
    total_pages = len(doc)
    st.write(f"読み込んだページ数: {len(page_texts)} / {total_pages}")

    # テキストをチャンクに分割
    page_chunks = split_text_into_chunks_by_page(page_texts)
    st.write(f"チャンク分割数: {len(page_chunks)}")

    st.subheader("チェック進捗状況")
    progress_bar = st.progress(0)
    check_status_text = st.empty()

    # 非同期チェック
    results, used_chunks, check_time = asyncio.run(run_async_check(client, page_chunks, progress_bar, check_status_text))

    # チェック結果をDataFrameにまとめる
    all_df_results = pd.DataFrame()
    for (page_num, result_str) in results:
        df_part = parse_json_results_to_dataframe(result_str, page_num)
        all_df_results = pd.concat([all_df_results, df_part], ignore_index=True)

    if not all_df_results.empty and "ページ番号" in all_df_results.columns:
        # 行番号・文字オフセット情報付加
        all_df_results = add_location_info(all_df_results, extracted_texts)
        all_df_results.sort_values(by=["ページ番号"], inplace=True)

        st.write("フィルタリング前のチェック結果:")
        st.dataframe(all_df_results.drop(columns=["重要度", "自信度", "情報種別"], errors='ignore'))
    else:
        st.write("指摘事項がありませんでした。")
        all_df_results = pd.DataFrame()  # 空にしておく

    # フィルタリング実行（非同期並列）
    if not all_df_results.empty:
        st.subheader("チェック結果フィルタリング中...")
        filtering_progress = st.progress(0)
        filtered_results_df, excluded_results_df, filtering_time = asyncio.run(run_async_filter(client, all_df_results, chunk_size=5, progress_bar=filtering_progress))

        if not filtered_results_df.empty and "ページ番号" in filtered_results_df.columns:
            filtered_results_df.sort_values(by=["ページ番号"], inplace=True)
        if not excluded_results_df.empty and "ページ番号" in excluded_results_df.columns:
            excluded_results_df.sort_values(by=["ページ番号"], inplace=True)

        st.write("フィルタリング後のチェック結果:")
        if filtered_results_df.empty:
            st.write("フィルタリング後に残った指摘事項はありません。")
        else:
            display_dataframe_with_copy(filtered_results_df)

        if not excluded_results_df.empty:
            st.write("除外された指摘事項:")
            st.dataframe(excluded_results_df.drop(columns=["重要度", "自信度", "情報種別"], errors='ignore'))
    else:
        filtered_results_df = pd.DataFrame()
        excluded_results_df = pd.DataFrame()
        filtering_time = 0.0

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    # 処理時間表示
    st.write(f"総処理時間: {total_processing_time:.2f} 秒")
    st.write(f"テキスト抽出処理時間: {extract_time:.2f} 秒")
    st.write(f"チェック処理時間: {check_time:.2f} 秒")
    st.write(f"フィルタリング処理時間: {filtering_time:.2f} 秒")

    # Excel出力用（不要列削除）
    excel_filtered = filtered_results_df.drop(columns=["重要度", "自信度", "情報種別"], errors='ignore') if not filtered_results_df.empty else pd.DataFrame()
    excel_excluded = excluded_results_df.drop(columns=["重要度", "自信度", "情報種別"], errors='ignore') if not excluded_results_df.empty else pd.DataFrame()
    excel_all = all_df_results.drop(columns=["重要度", "自信度", "情報種別"], errors='ignore') if not all_df_results.empty else pd.DataFrame()

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        if not excel_filtered.empty:
            excel_filtered.to_excel(writer, sheet_name='フィルタ後チェック結果', index=False)
        if not excel_excluded.empty:
            excel_excluded.to_excel(writer, sheet_name='除外された指摘項目', index=False)
        page_times_df = pd.DataFrame(page_processing_times)
        page_times_df.to_excel(writer, sheet_name='ページ処理時間', index=False)
        summary_df = pd.DataFrame({
            '合計処理時間（秒）': [total_processing_time],
            'テキスト抽出時間（秒）': [extract_time],
            'チェック処理時間（秒）': [check_time],
            'フィルタリング処理時間（秒）': [filtering_time]
        })
        summary_df.to_excel(writer, sheet_name='処理時間サマリ', index=False)
        extracted_texts_df = pd.DataFrame(extracted_texts)
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

else:
    st.write("PDFファイルをアップロードしてください。")
