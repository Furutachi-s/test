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
from functools import partial

# ===== 環境変数からAPIキーとエンドポイントを取得 =====
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not key or not endpoint:
    st.error("Azure OpenAIのAPIキーまたはエンドポイントが設定されていません。")

# ===== Azure OpenAI クライアント初期化 =====
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

chat_history = []
page_processing_times = []
extracted_texts = []
chat_logs = []

def preprocess_text(text):
    text = unicodedata.normalize('NFKC', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in ['\n', '\t'])
    text = re.sub(r'[ \t]+', ' ', text)
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
    st.write(f"PDFの総ページ数: {total_pages}")
    
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
            actual_page_num = page_num + 1  # 1から始まるページ番号
            page_texts.append((actual_page_num, text_content))
            processing_time = time.time() - page_start_time
            page_processing_times.append({'ページ番号': actual_page_num, '処理時間（秒）': processing_time})
            extracted_texts.append({'ページ番号': actual_page_num, 'テキスト': text_content})
            st.write(f"ページ {actual_page_num} のテキストを抽出しました。")
        except Exception as e:
            st.write(f"ページ {page_num + 1} の処理中にエラーが発生しました: {e}")
            page_texts.append((page_num + 1, ""))
            page_processing_times.append({'ページ番号': page_num + 1, '処理時間（秒）': 0})
            extracted_texts.append({'ページ番号': page_num + 1, 'テキスト': ""})
    return page_texts

def split_text_into_chunks_by_page(page_texts, chunk_size=2000, chunk_overlap=200):
    try:
        encoding = tiktoken.encoding_for_model('gpt-4o')  # モデル名を'gpt-4o'に修正
    except:
        encoding = tiktoken.get_encoding('cl100k_base')  # 代替エンコーディング
    page_chunks = []
    for page_num, text in page_texts:
        tokens = encoding.encode(text)
        for i in range(0, len(tokens), chunk_size - chunk_overlap):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = encoding.decode(chunk_tokens)
            page_chunks.append((page_num, chunk_text))
    return page_chunks

async def async_check_text_with_openai(client, text, page_num, semaphore):
    async with semaphore:
        await asyncio.sleep(0.5)  # 実際のAPI呼び出しに置き換えてください
        session_message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
あなたには建設機械のマニュアルをチェックしてもらいます（該当ページ: {page_num}）。
以下の項目についてチェックし、指摘してください: 
(1) 誤字、脱字
(2) 文脈的に間違っている内容
(3) 文法が間違っていて誤解を与える箇所

ただし以下は指摘しない:
- 不自然な改行や空白
- カタカナ単語末尾長音省略（例：バッテリ、モータ）

結果は以下フォーマットのJSONのみで返す（他の説明不要）:
[
  {{
    "page": {page_num},
    "category": "誤字" または "文法" または "文脈",
    "reason": "指摘理由を簡潔に",
    "error_location": "指摘箇所",
    "context": "周辺テキスト",
    "suggestion": "修正案を簡潔に",
    "importance": 整数1～5,
    "confidence": 整数1～5
  }},
  ...
]

以下のテキストをチェック：
{text}
"""}
        ]

        e = None
        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",  # モデル名を'gpt-4o'に修正
                    seed=42,
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                chat_history.append({
                    "page_num": page_num,
                    "messages": session_message,
                    "response": response.choices[0].message.content
                })
                st.write(f"ページ {page_num} のチェックが完了しました。")
                return response.choices[0].message.content
            except Exception as ex:
                e = ex
                st.write(f"ページ {page_num} のチェック中にエラーが発生しました。再試行します... ({ex})")
                await asyncio.sleep(5)

        chat_history.append({
            "page_num": page_num,
            "messages": session_message,
            "response": f"エラー: {e}" if e else "不明なエラー"
        })
        return ""

async def run_async_check(client, page_chunks, progress_bar, check_status_text):
    non_empty_chunks = [(p, t) for p, t in page_chunks if t.strip()]
    semaphore = asyncio.Semaphore(5)
    
    # タスク生成
    tasks = [
        async_check_text_with_openai(client, chunk_text, page_num, semaphore)
        for (page_num, chunk_text) in non_empty_chunks
    ]
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        check_status_text.write("チェック対象がありません。")
        return [], non_empty_chunks
    
    # gatherで順序を保持したまま結果取得
    results = await asyncio.gather(*tasks)
    
    # 進捗バー更新とステータス表示
    progress_bar.progress(1.0)
    check_status_text.write("チェックが完了しました！")
    
    return results, non_empty_chunks


def parse_json_results_to_dataframe(results, page_num):
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
            "修正案": error.get("suggestion", ""),
            "周辺テキスト": error.get("context", ""),
            "重要度": error.get("importance", ""),
            "自信度": error.get("confidence", "")
        }
        if any(issue.values()):
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

def filter_check_results_with_openai(df_results, chunk_size=5, progress_bar=None):
    filtered_results_list = []
    excluded_results_list = []
    total_results = len(df_results)
    if total_results == 0:
        if progress_bar:
            progress_bar.progress(1.0)
        return pd.DataFrame(), pd.DataFrame()
    num_chunks = (total_results // chunk_size) + (1 if total_results % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_results)
        df_chunk = df_results.iloc[start_idx:end_idx]

        if df_chunk.empty:
            if progress_bar:
                progress = (i + 1) / num_chunks
                progress_bar.progress(progress)
            continue

        session_message = []
        system_prompt = "You are a helpful assistant."
        session_message.append({"role": "system", "content": system_prompt})

        check_results_json = df_chunk.to_dict(orient='records')
        check_results_str = json.dumps(check_results_json, ensure_ascii=False, indent=2)

        prompt = f"""
以下のチェック結果リストがあります。

{check_results_str}

以下の項目は指摘しないでください(除外対象):
(1) 不自然な改行や空白
(2) カタカナ単語末尾の長音省略指摘（例：バッテリ、モータ末尾長音なしはOK）

純粋なJSONで以下形式で出力:
- "filtered_results": 除外でない結果リスト
- "excluded_results": 除外した結果リスト
"""
        session_message.append({"role": "user", "content": prompt})
        chat_logs.append({
            "filtering_chunk": i + 1,
            "messages": session_message.copy()
        })

        e = None
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # モデル名を'gpt-4o'に修正
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                chat_history.append({
                    "filtering_chunk": i + 1,
                    "messages": session_message,
                    "response": response.choices[0].message.content
                })
                chat_logs[-1]["response"] = response.choices[0].message.content

                filtered_chunk, excluded_chunk = parse_filter_response(response.choices[0].message.content)
                filtered_results_list.extend(filtered_chunk)
                excluded_results_list.extend(excluded_chunk)
                st.write(f"フィルタリングチャンク {i + 1} が完了しました。")
                break
            except Exception as ex:
                st.write(f"フィルタリングチャンク {i + 1} 中にエラーが発生しました。再試行します... ({ex})")
                time.sleep(5)
        else:
            st.write(f"フィルタリングチャンク {i + 1} でエラーが続いたため、スキップします。")
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

        if progress_bar:
            progress = (i + 1) / num_chunks
            progress_bar.progress(progress)

    filtered_results_df = pd.DataFrame(filtered_results_list)
    excluded_results_df = pd.DataFrame(excluded_results_list)
    return filtered_results_df, excluded_results_df

st.title("文章AIチェックシステム_非同期＋フィルタリング")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if uploaded_file is not None:
    st.write("PDFファイルがアップロードされました")
    total_start_time = time.time()

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    page_texts = extract_text_excluding_images_and_header_footer(doc)
    total_pages = len(doc)
    st.write(f"読み込んだページ数: {len(page_texts)} / {total_pages}")

    page_chunks = split_text_into_chunks_by_page(page_texts)
    st.write(f"チャンク分割数: {len(page_chunks)}")

    st.subheader("チェック進捗状況")
    progress_bar = st.progress(0)
    check_status_text = st.empty()

    # 非同期実行
    results, used_chunks = asyncio.run(run_async_check(client, page_chunks, progress_bar, check_status_text))

    # 結果解析
    all_df_results = pd.DataFrame()
    for (page_num, _), result_str in zip(used_chunks, results):
        df_part = parse_json_results_to_dataframe(result_str, page_num)
        all_df_results = pd.concat([all_df_results, df_part], ignore_index=True)

    st.write("フィルタリング前のチェック結果:")
    if all_df_results.empty:
        st.write("指摘事項がありませんでした。")
    else:
        st.dataframe(all_df_results)

    # フィルタリング実行
    st.subheader("チェック結果フィルタリング中...")
    filtering_progress = st.progress(0)
    filtered_results_df, excluded_results_df = filter_check_results_with_openai(all_df_results, chunk_size=5, progress_bar=filtering_progress)

    st.write("フィルタリング後のチェック結果:")
    if filtered_results_df.empty:
        st.write("フィルタリング後に残った指摘事項はありません。")
    else:
        st.dataframe(filtered_results_df)

    # 除外された指摘事項を表示
    if not excluded_results_df.empty:
        st.write("除外された指摘事項:")
        st.dataframe(excluded_results_df)

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    st.write(f"総処理時間: {total_processing_time:.2f} 秒")

    # 結果をExcel出力
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        filtered_results_df.to_excel(writer, sheet_name='フィルタ後チェック結果', index=False)
        if not excluded_results_df.empty:
            excluded_results_df.to_excel(writer, sheet_name='除外された指摘項目', index=False)
        page_times_df = pd.DataFrame(page_processing_times)
        page_times_df.to_excel(writer, sheet_name='ページ処理時間', index=False)
        summary_df = pd.DataFrame({'合計処理時間（秒）': [total_processing_time]})
        summary_df.to_excel(writer, sheet_name='合計処理時間', index=False)
        extracted_texts_df = pd.DataFrame(extracted_texts)
        extracted_texts_df.to_excel(writer, sheet_name='抽出テキスト', index=False)
        chat_logs_df = pd.DataFrame(chat_logs)
        chat_logs_df.to_excel(writer, sheet_name='チャットログ', index=False)
    output.seek(0)

    st.download_button(
        label="結果をExcelでダウンロード",
        data=output.getvalue(),
        file_name="チェック結果_フィルタリングあり.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.write("PDFファイルをアップロードしてください。")
