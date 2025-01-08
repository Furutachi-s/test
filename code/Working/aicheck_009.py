import os
import asyncio
import re
import time
import json
import unicodedata
from io import BytesIO
from typing import List, Tuple, Dict, Any

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from openai import AzureOpenAI
import tiktoken

# --- Streamlit設定 ---
st.set_page_config(layout="wide")

# --- セッションステート初期化関数 ---
def initialize_session_state():
    """セッションステートの初期化を行う。"""
    initial_states = {
        "processing_done": False,
        "uploaded_pdf_data": None,
        "extracted_texts": [],
        "page_processing_times": [],
        "chat_logs": [],
        "all_df_results": pd.DataFrame(),
        "filtered_results_df": pd.DataFrame(),
        "excluded_results_df": pd.DataFrame(),
        "extract_time": 0.0,
        "check_time": 0.0,
        "filtering_time": 0.0,
        "total_processing_time": 0.0,
        "dictionary_words": set(),
    }
    for key, value in initial_states.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# --- Azure OpenAI認証情報の設定 ---
def get_openai_client() -> AzureOpenAI:
    """Azure OpenAIのクライアントを取得する。"""
    key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not key or not endpoint:
        st.error("Azure OpenAIのAPIキーまたはエンドポイントが設定されていません。")
        st.stop()

    return AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

client = get_openai_client()

# --- テキスト前処理 ---
def preprocess_text(text: str) -> str:
    """テキストの前処理を行う。"""
    text = unicodedata.normalize('NFKC', text)
    # 制御文字削除（\n, \t は残す）
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in ['\n', '\t'])
    # 連続する空白を1つにまとめる
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# --- 矩形オーバーラップ判定 ---
def rects_overlap(rect1: List[float], rect2: List[float]) -> bool:
    """2つの矩形が重なっているか判定する。"""
    x0_1, y0_1, x1_1, y1_1 = rect1
    x0_2, y0_2, x1_2, y1_2 = rect2
    return not (x1_1 <= x0_2 or x1_2 <= x0_1 or y1_1 <= y0_2 or y1_2 <= y0_1)

# --- ヘッダ/フッタ判定 ---
def is_in_header_footer(block_bbox: List[float], page_height: float, header_height: float, footer_height: float) -> bool:
    """ブロックがヘッダまたはフッタに含まれるか判定する。"""
    _, y0, _, y1 = block_bbox
    return y0 <= header_height or y1 >= page_height - footer_height

# --- 画像とヘッダ・フッタを除外してテキスト抽出 ---
def extract_text_excluding_images_and_header_footer(pdf_document: fitz.Document) -> Tuple[List[Tuple[int, str]], float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """画像とヘッダ・フッタを除外してPDFからテキストを抽出する。"""
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
                if block["type"] != 0:
                    continue
                block_bbox = block["bbox"]
                if any(rects_overlap(block_bbox, ib) for ib in image_bboxes):
                    continue
                if is_in_header_footer(block_bbox, page_height, header_height, footer_height):
                    continue
                for line in block["lines"]:
                    # スパン間を半角スペースで結合
                    line_text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
                    line_text = preprocess_text(line_text)
                    if line_text:
                        text_content += line_text + "\n"
                text_content += "\n"

            text_content = preprocess_text(text_content)
            actual_page_num = page_num + 1
            page_texts.append((actual_page_num, text_content))

            processing_time = time.time() - page_start_time
            page_processing_times_local.append({'ページ番号': actual_page_num, '処理時間（秒）': processing_time})
            extracted_texts_local.append({'ページ番号': actual_page_num, 'テキスト': text_content})

            # 進捗更新
            extract_progress_bar.progress((page_num + 1) / total_pages)
            extract_status_text.text(f"テキスト抽出中... {page_num + 1}/{total_pages}")

        except Exception as e:
            st.write(f"ページ {page_num + 1} の処理中にエラーが発生しました: {e}")
            page_texts.append((page_num + 1, ""))
            page_processing_times_local.append({'ページ番号': page_num + 1, '処理時間（秒）': 0})
            extracted_texts_local.append({'ページ番号': page_num + 1, 'テキスト': ""})

    extract_end_time = time.time()
    extract_time = extract_end_time - extract_start_time
    extract_status_text.text("テキスト抽出完了")
    st.write(f"テキスト抽出処理時間: {extract_time:.2f} 秒")
    return page_texts, extract_time, page_processing_times_local, extracted_texts_local

# --- テキストをチャンクに分割 ---
def split_text_into_chunks_by_page(page_texts: List[Tuple[int, str]], chunk_size: int = 2000, chunk_overlap: int = 200) -> List[Tuple[int, str]]:
    """ページごとのテキストをチャンクに分割する。"""
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

# --- JSON文字列をDataFrameに変換 ---
def parse_json_results_to_dataframe(results: str, page_num: int) -> pd.DataFrame:
    """JSON形式の結果をDataFrameに変換する。"""
    if not results.strip():
        return pd.DataFrame()

    # コードブロックの除去
    text = re.sub(r'^```(?:json)?', '', results.strip(), flags=re.MULTILINE)
    text = re.sub(r'```$', '', text, flags=re.MULTILINE)
    text = text.strip()

    try:
        json_results = json.loads(text)
    except json.JSONDecodeError:
        st.write(f"ページ {page_num} のJSON解析に失敗しました。返却値: {text[:200]}...")
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
            "重要度": error.get("importance", ""),
            "検出元指示": error.get("prompt_source", "")
        }
        if any(val for val in issue.values()):
            issues.append(issue)

    df = pd.DataFrame(issues)
    if not df.empty:
        df.replace("", pd.NA, inplace=True)
        df.dropna(how='all', inplace=True)
    return df

# --- AIフィルタリング結果のパース ---
def parse_filter_response(response_text: str) -> Tuple[List[Any], List[Any]]:
    """フィルタリング結果のJSONをパースする。"""
    response_text = response_text.strip()
    response_text = re.sub(r'^```(?:json)?', '', response_text, flags=re.MULTILINE)
    response_text = re.sub(r'```$', '', response_text, flags=re.MULTILINE)
    response_text = response_text.strip()

    match = re.search(r'(\{.*\}|\[.*\])', response_text, re.DOTALL)
    json_str = match.group(0) if match else response_text

    try:
        json_data = json.loads(json_str)
    except json.JSONDecodeError:
        return [], []

    filtered_results = json_data.get('filtered_results', [])
    excluded_results = json_data.get('excluded_results', [])
    return filtered_results, excluded_results

# --- OpenAI API呼び出し(非同期): チェック ---
async def async_check_text_with_openai(client: AzureOpenAI, text: str, page_num: int, semaphore: asyncio.Semaphore, chat_logs_local: List[Dict[str, Any]]) -> Tuple[int, str]:
    """OpenAI APIを非同期で呼び出し、テキストのチェックを行う。"""
    system_prompt = """あなたは建設機械マニュアルのチェックアシスタントです。以下の事項を確認し、資料の品質をチェックしてください。

【指摘してほしい問題点】
(1) 誤字や脱字
(2) 意味伝達に支障がある文法ミス
(3) 文脈的に誤った記載内容

【追加で行うこと】
- 修正しないと重大な誤解を与える恐れがあるもの（重要度: "高"）をマークする
- 修正しなくても意味伝達に支障がないもの（重要度: "低"）をマークする

【分析用要件】
- どの指示に従って検出したかを "prompt_source" フィールドに記載してください。
  (例: "誤字や脱字の指示に基づく" / "文脈的に誤った記載の指示に基づく" など)

【指摘不要】
- 不自然な改行・空白（PDF抽出由来のため）
- カタカナ用語の末尾長音の有無に関する指摘

【回答形式】  
- 出力は必ず **JSONのみ** を返し、それ以外の文字列（説明文など）は一切出力しないでください。
- 問題がなければ空の配列 `[]` のみを返してください。

【JSONフォーマット例】
[
  {
    "page": <page_num>,
    "category": "誤字" or "文法" or "文脈",
    "reason": "具体的な理由",
    "error_location": "指摘箇所",
    "suggestion": "修正案",
    "importance": "高" or "低",
    "prompt_source": "どの指示による検出なのかを簡潔に"
  }
]
"""

    user_prompt = f"以下は、PDF抽出テキストの一部です（該当ページ: {page_num}）。\n\n{text}"
    session_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    async with semaphore:
        for attempt in range(3):
            try:
                await asyncio.sleep(1.0)  # レート制限対策
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                assistant_response = response.choices[0].message.content
                chat_logs_local.append({
                    "page_num": page_num,
                    "messages": session_message,
                    "response": assistant_response
                })
                return page_num, assistant_response
            except Exception as ex:
                if attempt < 2:
                    st.write(f"ページ {page_num} のチェック中にエラーが発生しました。再試行します... ({ex})")
                    await asyncio.sleep(5)
                else:
                    st.warning(f"ページ {page_num} のチェックに失敗しました。")
                    chat_logs_local.append({
                        "page_num": page_num,
                        "messages": session_message,
                        "response": f"エラー: {ex}"
                    })
                    return page_num, ""

# --- OpenAI API呼び出し(非同期): フィルタリング ---
async def async_filter_chunk_with_openai(client: AzureOpenAI, df_chunk: pd.DataFrame, chunk_index: int, semaphore: asyncio.Semaphore, chat_logs_local: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    """OpenAI APIを非同期で呼び出し、チェック結果をフィルタリングする。"""
    system_prompt = """あなたはチェック結果を整理するフィルタリングアシスタントです。
以下の方針で、チェック結果をふるい分けてください。

【excluded_results に回す項目】  
- 不自然な改行や空白に関する指摘
- 指摘箇所と修正案の違いが、カタカナ用語の長音符に関する違いのみの場合
- 指摘箇所と修正案が全く同じ項目

【filtered_results に残す項目】  
- 上記以外の誤字・文法・文脈ミス

【追加要件】
- filtered_results の各項目には "importance_reason" を付与し、
  なぜ重要度("高"/"低")と判断したのかを短く説明してください。
- excluded_results の各項目には "excluded_reason" を付与し、
  なぜ除外したのかを短く説明してください。

必ず以下形式のJSONのみを出力してください:
{
  "filtered_results": [...],
  "excluded_results": [...]
}
"""

    check_results_json = df_chunk.to_dict(orient='records')
    check_results_str = json.dumps(check_results_json, ensure_ascii=False, indent=2)
    user_prompt = f"以下はチェック結果のリストです。\n\n{check_results_str}\n\n上記ルールに基づき、\"filtered_results\"と\"excluded_results\"に振り分けてください。"

    session_message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    async with semaphore:
        for attempt in range(3):
            try:
                await asyncio.sleep(0.5)  # レート制限対策
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    temperature=0,
                    max_tokens=2000,
                    messages=session_message
                )
                assistant_response = response.choices[0].message.content
                chat_logs_local.append({
                    "chunk_index": chunk_index,
                    "messages": session_message,
                    "response": assistant_response
                })
                filtered, excluded = parse_filter_response(assistant_response)
                return filtered, excluded
            except Exception as ex:
                if attempt < 2:
                    st.write(f"フィルタリングチャンク {chunk_index} の処理中にエラーが発生しました。再試行します... ({ex})")
                    await asyncio.sleep(5)
                else:
                    st.warning(f"フィルタリングチャンク {chunk_index} の処理に失敗しました。")
                    chat_logs_local.append({
                        "chunk_index": chunk_index,
                        "messages": session_message,
                        "response": f"エラー: {ex}"
                    })
                    return [], []

# --- OpenAI API呼び出し(非同期): チェック実行 ---
async def run_async_check(client: AzureOpenAI, page_chunks: List[Tuple[int, str]], progress_bar: st.progress, check_status_text: st.empty, chat_logs_local: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], float]:
    """非同期でチェックを実行する。"""
    non_empty_chunks = [(p, t) for p, t in page_chunks if t.strip()]
    semaphore = asyncio.Semaphore(5)
    
    tasks = [
        asyncio.create_task(
            async_check_text_with_openai(client, chunk_text, page_num, semaphore, chat_logs_local)
        )
        for (page_num, chunk_text) in non_empty_chunks
    ]
    
    total_tasks = len(tasks)
    if total_tasks == 0:
        check_status_text.write("チェック対象がありません。")
        return [], [], 0.0

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
    check_time_local = check_end_time - check_start_time
    check_status_text.text(f"チェックが完了しました！ (処理時間: {check_time_local:.2f} 秒)")
    progress_bar.progress(1.0)

    return results, non_empty_chunks, check_time_local

# --- フィルタリング実行(非同期) ---
async def run_async_filter(client: AzureOpenAI, df_results: pd.DataFrame, chunk_size: int, progress_bar: st.progress, chat_logs_local: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """非同期でフィルタリングを実行する。"""
    filtered_results_list = []
    excluded_results_list = []
    total_results = len(df_results)
    if total_results == 0:
        progress_bar.progress(1.0)
        return pd.DataFrame(), pd.DataFrame(), 0.0

    filtering_start_time = time.time()
    num_chunks = (total_results // chunk_size) + (1 if total_results % chunk_size != 0 else 0)

    filter_status_text = st.empty()
    filter_status_text.text("フィルタリング中...")
    progress_bar.progress(0.0)

    semaphore = asyncio.Semaphore(5)
    tasks = [
        asyncio.create_task(
            async_filter_chunk_with_openai(client, df_results.iloc[i*chunk_size:(i+1)*chunk_size], i+1, semaphore, chat_logs_local)
        )
        for i in range(num_chunks)
    ]

    completed = 0
    for task in asyncio.as_completed(tasks):
        f_chunk, e_chunk = await task
        filtered_results_list.extend(f_chunk)
        excluded_results_list.extend(e_chunk)
        completed += 1
        progress = completed / num_chunks
        progress_bar.progress(progress)
        filter_status_text.text(f"フィルタリング中... {completed}/{num_chunks} チャンク完了")

    filtering_end_time = time.time()
    filtering_time_local = filtering_end_time - filtering_start_time
    filter_status_text.text(f"フィルタリング完了 (処理時間: {filtering_time_local:.2f} 秒)")

    filtered_results_df = pd.DataFrame(filtered_results_list)
    excluded_results_df = pd.DataFrame(excluded_results_list)

    return filtered_results_df, excluded_results_df, filtering_time_local

# --- 指摘箇所の行番号・文字オフセット情報を付与 ---
def add_location_info(df: pd.DataFrame, extracted_texts: List[Dict[str, Any]]) -> pd.DataFrame:
    """指摘箇所に行番号と文字オフセット情報を付与する。"""
    page_map = {x["ページ番号"]: x["テキスト"] for x in extracted_texts}

    df["行番号"] = None
    df["文字オフセット"] = None

    for i, row in df.iterrows():
        page_num = row.get("ページ番号")
        err_loc = row.get("指摘箇所")
        if pd.isna(err_loc) or page_num not in page_map:
            continue

        err_loc = unicodedata.normalize('NFKC', str(err_loc)).strip()
        full_text = unicodedata.normalize('NFKC', page_map[page_num]).strip()

        idx = full_text.find(err_loc)

        if idx == -1:
            # 近似検索（スペース区切り）
            words = err_loc.split()
            candidates = [full_text.find(w) for w in words if w and full_text.find(w) != -1]
            idx = min(candidates) if candidates else -1

        if idx == -1:
            continue

        line_number = full_text.count('\n', 0, idx) + 1
        df.at[i, "行番号"] = line_number
        df.at[i, "文字オフセット"] = f"{idx}文字目付近"

    return df

# --- DataFrame表示ラッパ ---
def display_dataframe(df: pd.DataFrame):
    """DataFrameを表示する。"""
    st.dataframe(df)

# --- チャットログを見やすい形式にする ---
def create_readable_chat_dataframe(chat_logs_local: List[Dict[str, Any]]) -> pd.DataFrame:
    """チャットログを見やすい形式に変換する。"""
    records = []
    for log in chat_logs_local:
        page_or_chunk = log.get("page_num") or log.get("chunk_index")
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

# --- JSONフィルタリング関数 (不要なので削除) ---
# --- ここでは関数を必要に応じて定義してください。 ---

# --- アプリタイトル ---
st.title("文章AIチェックシステム_調整版")

# --- PDFアップロードUI ---
uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

# --- 辞書ファイルアップロードUI ---
dictionary_file = None
if st.session_state.uploaded_pdf_data is None and uploaded_file is not None:
    st.write("【任意】辞書用Excelファイル(除外用ワード一覧)をアップロードする場合はA列にワードを入れてください:")
    dictionary_file = st.file_uploader("辞書ファイル(Excel形式)", type=["xlsx", "xls"])

# --- PDF & 辞書ファイル読み込み ---
if uploaded_file is not None and st.session_state.uploaded_pdf_data is None:
    # PDF読み込み
    st.session_state.uploaded_pdf_data = uploaded_file.read()
    # 辞書ファイル読み込み
    if dictionary_file is not None:
        try:
            dict_df = pd.read_excel(dictionary_file, sheet_name=0, header=None)
            dict_df = dict_df.dropna(subset=[0])
            st.session_state.dictionary_words = set(dict_df[0].astype(str).str.strip().tolist())
            st.success("辞書ファイルを正常に読み込みました。")
        except Exception as e:
            st.error(f"辞書ファイルの読み込みに失敗しました: {e}")

# --- メイン処理: PDFデータがあり、まだ処理していない場合 ---
if st.session_state.uploaded_pdf_data is not None and not st.session_state.processing_done:
    total_start_time = time.time()

    # PDFを読み込み
    try:
        doc = fitz.open(stream=st.session_state.uploaded_pdf_data, filetype="pdf")
    except Exception as e:
        st.error(f"PDFの読み込みに失敗しました: {e}")
        st.stop()

    # テキスト抽出
    page_texts, extract_time_local, page_processing_times_local, extracted_texts_local = extract_text_excluding_images_and_header_footer(doc)
    total_pages = len(doc)
    st.write(f"読み込んだページ数: {len(page_texts)} / {total_pages}")

    # セッションステートに保存
    st.session_state.page_processing_times = page_processing_times_local
    st.session_state.extracted_texts = extracted_texts_local
    st.session_state.extract_time = extract_time_local

    # チャンク分割
    page_chunks = split_text_into_chunks_by_page(page_texts)
    st.write(f"チャンク分割数: {len(page_chunks)}")

    # チェック進捗表示
    st.subheader("チェック進捗状況")
    check_progress_bar = st.progress(0)
    check_status_text = st.empty()

    # 非同期チェック実行
    chat_logs_local = []
    try:
        results, used_chunks, check_time_local = asyncio.run(
            run_async_check(client, page_chunks, check_progress_bar, check_status_text, chat_logs_local)
        )
    except Exception as e:
        st.error(f"チェック処理中にエラーが発生しました: {e}")
        st.stop()

    st.session_state.chat_logs = chat_logs_local
    st.session_state.check_time = check_time_local

    # チェック結果を統合
    all_df_results_local = pd.DataFrame()
    for page_num, result_str in results:
        df_part = parse_json_results_to_dataframe(result_str, page_num)
        all_df_results_local = pd.concat([all_df_results_local, df_part], ignore_index=True)

    # 重複削除
    if not all_df_results_local.empty:
        duplicate_key_columns = ["ページ番号", "指摘箇所", "指摘理由", "修正案"]
        all_df_results_local.drop_duplicates(subset=duplicate_key_columns, keep='first', inplace=True)

    # 行番号付加
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
        try:
            filtered_results_df_local, excluded_results_df_local, filtering_time_local = asyncio.run(
                run_async_filter(client, all_df_results_local, chunk_size=5, progress_bar=filtering_progress, chat_logs_local=st.session_state.chat_logs)
            )
        except Exception as e:
            st.error(f"フィルタリング処理中にエラーが発生しました: {e}")
            filtered_results_df_local, excluded_results_df_local, filtering_time_local = pd.DataFrame(), pd.DataFrame(), 0.0

        # AIフィルタリングによる除外
        if not excluded_results_df_local.empty:
            excluded_results_df_local["除外理由"] = "AIフィルタリングによる"

        # 辞書による除外
        if not filtered_results_df_local.empty and st.session_state.dictionary_words:
            in_dict = filtered_results_df_local["指摘箇所"].isin(st.session_state.dictionary_words)
            dict_excluded = filtered_results_df_local[in_dict].copy()
            if not dict_excluded.empty:
                dict_excluded["除外理由"] = "辞書により除外"
                excluded_results_df_local = pd.concat([excluded_results_df_local, dict_excluded], ignore_index=True)
                filtered_results_df_local = filtered_results_df_local[~in_dict]

        # ソート
        if not filtered_results_df_local.empty and "ページ番号" in filtered_results_df_local.columns:
            filtered_results_df_local.sort_values(by=["ページ番号"], inplace=True)
        if not excluded_results_df_local.empty and "ページ番号" in excluded_results_df_local.columns:
            excluded_results_df_local.sort_values(by=["ページ番号"], inplace=True)

        # 結果表示
        st.write("フィルタリング後のチェック結果:")
        if filtered_results_df_local.empty:
            st.write("フィルタリング後に残った指摘事項はありません。")
        else:
            display_dataframe(filtered_results_df_local)

        if not excluded_results_df_local.empty:
            st.write("除外された指摘事項:")
            st.dataframe(excluded_results_df_local)

    else:
        filtered_results_df_local = pd.DataFrame()
        excluded_results_df_local = pd.DataFrame()
        filtering_time_local = 0.0

    # セッションステートに保存
    st.session_state.filtered_results_df = filtered_results_df_local
    st.session_state.excluded_results_df = excluded_results_df_local
    st.session_state.filtering_time = filtering_time_local

    total_end_time = time.time()
    total_processing_time_local = total_end_time - total_start_time
    st.session_state.total_processing_time = total_processing_time_local

    # 処理時間表示
    st.write(f"総処理時間: {total_processing_time_local:.2f} 秒")
    st.write(f"テキスト抽出処理時間: {extract_time_local:.2f} 秒")
    st.write(f"チェック処理時間: {check_time_local:.2f} 秒")
    st.write(f"フィルタリング処理時間: {filtering_time_local:.2f} 秒")

    st.session_state.processing_done = True

# --- 処理完了後の表示・Excelダウンロード ---
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
