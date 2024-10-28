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

# 環境変数からAPIキーとエンドポイントを取得
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

# チャット形式でOpenAIに問い合わせる関数
def check_text_with_openai(text, pages):
    session_message = []
    session_message.append({"role": "system", "content": "You are a helpful assistant."})

    prompt = f"""
あなたには建設機械のマニュアルをチェックしてもらいます（該当ページ: {pages}）。
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

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用するモデル
        seed=42,
        temperature=0,
        max_tokens=4000,  # トークン数を調整
        messages=session_message
    )

    return response.choices[0].message.content

# 画像をBase64形式でエンコードして埋め込む関数
def encode_image_to_base64(image):
    image_stream = BytesIO()
    image.save(image_stream, format="PNG")  # PNG形式で保存
    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')
    return f"![Embedded Image](data:image/png;base64,{image_base64})"

# PyMuPDF4LLMを使用してPDFを解析し、テキスト、表、画像を含むマークダウン形式でテキストを抽出する関数
def extract_text_with_pymupdf4llm(pdf_document):
    # PDFをマークダウン形式で抽出
    markdown_text = to_markdown(pdf_document)
    
    return markdown_text

# テキストをトークン数に基づいてチャンクに分割する関数
def split_text_into_chunks(text, chunk_size=2000, chunk_overlap=200):
    encoding = tiktoken.encoding_for_model('gpt-4')
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

# チャンクが意図通り分割されているか確認するデバッグ表示
def display_chunks_debug(chunks):
    st.subheader("チャンク内容のデバッグ表示")
    for i, chunk_text in enumerate(chunks):
        st.write(f"チャンク {i+1} の内容:")
        st.write(chunk_text[:100] + "...")  # 最初の100文字を表示
        st.write("---")

# チェック結果をパースしてDataFrameに変換する関数
def parse_json_results_to_dataframe(results, pages):
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
            "ページ番号": ", ".join(map(str, pages)),
            "カテゴリ": error.get("category", ""),  # カテゴリの追加
            "指摘箇所": error.get("error_location", ""),
            "指摘理由": error.get("reason", ""),  # 指摘理由を取得
            "周辺テキスト": error.get("context", "")
        }

        if issue["指摘箇所"] or issue["指摘理由"] or issue["周辺テキスト"]:
            issues.append(issue)

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

    # PDFからテキストをマークダウン形式で抽出
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    markdown_text = extract_text_with_pymupdf4llm(doc)

    # チャンクに分割
    chunks = split_text_into_chunks(markdown_text)

    # デバッグ: チャンクが意図通り分割されているか確認
    display_chunks_debug(chunks)

    # 各チャンクごとにAzure OpenAIでテキストチェック
    all_df_results = pd.DataFrame()
    for i, chunk_text in enumerate(chunks):
        st.write(f"チャンク {i+1}/{len(chunks)} をチェックしています...")

        # チャンクが空でないことを確認
        if chunk_text.strip():
            check_result = check_text_with_openai(chunk_text, [i+1])
            df_results = parse_json_results_to_dataframe(check_result, [i+1])
            all_df_results = pd.concat([all_df_results, df_results], ignore_index=True)
        else:
            st.write(f"チャンク {i+1} は空なのでスキップされました。")

    # チェック結果を表示（テーブル形式で見やすく）
    st.subheader("AIによるチェック結果")

    if all_df_results.empty:
        st.write("チェック結果が空です")
    else:
        st.dataframe(all_df_results)

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
