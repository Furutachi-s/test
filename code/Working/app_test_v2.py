import os
import streamlit as st
import fitz  # PyMuPDFとしてインポート
from openai import AzureOpenAI
import pandas as pd
from io import BytesIO
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 環境変数からAPIキーとエンドポイントを取得
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

# チャット形式でOpenAIに問い合わせる関数
def check_text_with_openai(text):
    session_message = []
    session_message.append({"role": "system", "content": "You are a helpful assistant."})
    
    # ユーザーのプロンプトを作成
    prompt = f"""
    あなたには建設機械のマニュアルをチェックしてもらいます。
    以下の項目についてチェックし、指摘してください: 
    (1) 誤字
    (2) 文脈的に、記載内容が明確に間違っている場合
    (3) 文法が致命的に間違っていて、文章として意味が通らない場合

    ただし、以下の項目については指摘しないこと:
    (1) 偽陽性を避けるため、判断に迷った場合は指摘しない
    (2) 不自然な空白、半角スペースはPDF抽出時の仕様のため、指摘しない

    回答形式:
    ・各指摘については、以下の形式でJSONとして返し、JSON以外の文字列を一切含めないでください。コードブロックや追加の説明も含めないでください。
    - "page": 該当ページ番号
    - "category": 指摘のカテゴリ（例: 誤字、文法、文脈）
    - "reason": 指摘した理由（簡潔に記述）
    - "error_location": 指摘箇所
    - "context": 周辺テキスト
    """
    
    session_message.append({"role": "user", "content": prompt})

    # Azure OpenAIにリクエストを送信
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用するモデル
        seed=42,
        temperature=0,
        max_tokens=4000,  # トークン数を調整
        messages=session_message
    )

    # 結果を返す
    return response.choices[0].message.content

# ページ単位のチャンクにさらに細かい分割を加える
def split_text_into_smaller_chunks(text, chunk_size=2000, chunk_overlap=200):
    # RecursiveCharacterTextSplitterを使用して、指定された文字数で再分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 1つのチャンクの最大文字数
        chunk_overlap=chunk_overlap  # チャンク間のオーバーラップ部分
    )
    return splitter.split_text(text)

# ページ内容をチャンクに分割し、さらに細かく分割する
def combine_pages_for_chunking(pdf_document, max_chars_per_chunk=10000, chunk_size=2000):
    combined_text = ""
    text_chunks = []
    current_chunk_length = 0
    page_tracker = []  # ページ番号を記録

    # 各ページからテキストを抽出し、チャンクに追加
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)  # 各ページをロード
        text = page.get_text("text")  # テキストを抽出
        
        if text.strip():  # 空でないページのみ処理
            if current_chunk_length + len(text) > max_chars_per_chunk:
                # 大きすぎる場合は、さらに細かく分割
                smaller_chunks = split_text_into_smaller_chunks(combined_text, chunk_size=chunk_size)
                text_chunks.extend([(chunk, page_tracker) for chunk in smaller_chunks])
                
                # 新しいチャンクを開始
                combined_text = text
                current_chunk_length = len(text)
                page_tracker = [page_num + 1]  # ページ番号をリセット
            else:
                combined_text += text
                current_chunk_length += len(text)
                page_tracker.append(page_num + 1)  # ページ番号を追跡

    # 残ったテキストも追加
    if combined_text:
        smaller_chunks = split_text_into_smaller_chunks(combined_text, chunk_size=chunk_size)
        text_chunks.extend([(chunk, page_tracker) for chunk in smaller_chunks])

    st.write(f"{len(text_chunks)} 個のチャンクに分割されました（最大文字数: {chunk_size}）。")
    return text_chunks

# 結果をパースしてDataFrameに変換する関数 (修正済み)
def parse_json_results_to_dataframe(results, pages):
    issues = []
    
    # 余計なフォーマットを除去し、不完全なJSONの修正
    results = results.strip().strip("```").strip("json").strip()  # バッククオートや"json"ラベルを削除
    
    # カンマがある場合、それを削除してからパースする
    results = results.replace(",\n}", "\n}")

    # JSON文字列を辞書にパース
    try:
        json_results = json.loads(results)
    except json.JSONDecodeError as e:
        st.write(f"JSONの解析に失敗しました: {e}")
        return pd.DataFrame()  # 解析失敗時には空のDataFrameを返す

    # json_resultsがリスト形式であることを想定して処理
    if isinstance(json_results, list):
        errors_list = json_results  # リストがそのまま指摘結果
    else:
        errors_list = json_results.get("errors", [])  # 万が一辞書形式の場合の処理

    # 各指摘結果をDataFrameに変換
    for error in errors_list:
        for page in pages:  # チャンクのページ番号ごとに処理
            issue = {
                "ページ番号": page,  # 各ページ番号ごとに記録
                "指摘箇所": error.get("error_location", ""),
                "指摘理由": error.get("reason", ""),
                "周辺テキスト": error.get("context", "")
            }

            # データが空でない場合のみ追加
            if issue["指摘箇所"] or issue["指摘理由"] or issue["周辺テキスト"]:
                issues.append(issue)

    # DataFrameに変換してから、空の行を削除
    df = pd.DataFrame(issues)
    
    if not df.empty:
        df.replace("", pd.NA, inplace=True)
        df.dropna(how='all', inplace=True)
    
    return df

# StreamlitでUIを作成
st.title("PDF文章チェックシステム")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if uploaded_file is not None:
    st.write("PDFファイルがアップロードされました")

    # PDFからチャンクに分割するテキストを抽出
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    # ページ内容を最大文字数に基づいてチャンクにまとめる
    chunks_with_pages = combine_pages_for_chunking(doc)

    # 全チャンクの結果を格納するDataFrame
    all_df_results = pd.DataFrame()

    # 各チャンクごとにAzure OpenAIでテキストチェック
    all_check_results = []
    for i, (chunk_text, pages) in enumerate(chunks_with_pages):
        # Debug: 各チャンクの内容を確認
        st.write(f"チャンク {i+1}/{len(chunks_with_pages)} の内容: {chunk_text[:100]}...")  # チャンクの最初の100文字のみ表示
        st.write(f"チャンク {i+1}/{len(chunks_with_pages)} をチェックしています... (対応するページ: {pages})")
        
        # チャンクが空でないことを確認
        if chunk_text.strip():
            check_result = check_text_with_openai(chunk_text)
            all_check_results.append(check_result)

            # 各チャンクの結果をDataFrameに変換して集約
            df_results = parse_json_results_to_dataframe(check_result, pages)
            all_df_results = pd.concat([all_df_results, df_results], ignore_index=True)
        else:
            st.write(f"チャンク {i+1} は空なのでスキップされました。")

    # デバッグ: 全てのチェック結果を結合して表示
    st.write("Debug: チェック結果（全チャンク、未パース）:")
    combined_check_result = "\n\n".join(all_check_results)
    st.text_area("チェック結果（未パース）", value=combined_check_result, height=200)

    # チェック結果を表示（テーブル形式で見やすく）
    st.subheader("AIによるチェック結果")
    
    # DataFrameで結果を表示（ページ番号ごとに表示）
    if all_df_results.empty:
        st.write("チェック結果が空です")
    else:
        st.dataframe(all_df_results)

    # データをExcelファイルに変換してダウンロード
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        all_df_results.to_excel(writer, index=False)

    processed_data = output.getvalue()  # getvalue() でExcelファイルの内容を取得

    # ダウンロードオプション
    st.download_button(
        label="結果をExcelでダウンロード",
        data=processed_data,
        file_name="チェック結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
