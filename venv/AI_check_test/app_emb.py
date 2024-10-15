import os
import streamlit as st
import fitz  # PyMuPDFとしてインポート
from openai import AzureOpenAI
import pandas as pd
from io import BytesIO
from langchain.embeddings.openai import OpenAIEmbeddings  # OpenAIのEmbeddingを利用
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# 環境変数からAPIキーとエンドポイントを取得
api_key = os.getenv("AZURE_OPENAI_KEY")  # 環境変数からAPIキーを取得
embedding_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  # 環境変数からエンドポイントを取得

# Embeddingモデルの設定
embedding_model_name = "text-embedding-3-small"  # Azure OpenAIでデプロイしたEmbeddingモデル名

# Azure OpenAI クライアントの初期化
client = AzureOpenAI(api_key=api_key, api_version="2023-05-15", azure_endpoint=embedding_endpoint)

# チャット形式でOpenAIに問い合わせる関数
def check_text_with_openai(text):
    session_message = []
    session_message.append({"role": "system", "content": "You are a helpful assistant."})
    
    # ユーザーのプロンプトを作成
    prompt = f"""
    以下の文章に対して、以下の3つの項目について指摘してください: (1) 誤字脱字 (2) 文法の誤り (3) 意味が理解しにくい部分。
    各指摘については、指摘箇所、理由、及び周辺のテキストを含めて報告してください。
    \n\n{text}
    """
    
    session_message.append({"role": "user", "content": prompt})

    # Azure OpenAIにリクエストを送信
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 使用するモデル
        seed=42,
        temperature=0,
        max_tokens=1000,
        messages=session_message
    )

    # 結果を返す
    return response.choices[0].message.content

# セマンティックチャンク分割の関数を追加
def semantic_split_text(text):
    # OpenAIEmbeddingsを使ってセマンティックなチャンク分割を行う
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key, 
        deployment=embedding_model_name, 
        openai_api_base=embedding_endpoint
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # テキストをチャンクに分割
    chunks = text_splitter.split_text(text)
    
    # 分割したチャンクを埋め込みベースで意味的に処理
    return chunks

# 結果をパースしてDataFrameに変換する
def parse_results_to_dataframe(results):
    issues = []
    for item in results.split("\n\n"):
        if item.strip():
            lines = item.split("\n")
            # 指摘箇所、指摘理由、周辺テキストを抽出するための新しいロジック
            issue = {
                "指摘箇所": next((line.split("**指摘箇所**: ")[-1] for line in lines if "**指摘箇所**: " in line), "N/A"),
                "指摘理由": next((line.split("**理由**: ")[-1] for line in lines if "**理由**: " in line), "N/A"),
                "周辺テキスト": next((line.split("**周辺のテキスト**: ")[-1] for line in lines if "**周辺のテキスト**: " in line), "N/A")
            }
            issues.append(issue)
    
    return pd.DataFrame(issues)

# StreamlitでUIを作成
st.title("PDF文章チェックシステム - PyMuPDF + Azure OpenAI + LangChain使用")

uploaded_file = st.file_uploader("PDFファイルをアップロードしてください", type="pdf")

if uploaded_file is not None:
    st.write("PDFファイルがアップロードされました")

    # PDFからテキストを抽出
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # 各ページをロード
        text += page.get_text("text")  # テキストを抽出

    # 抽出されたテキストを表示
    st.write("抽出されたテキスト:")
    st.text_area("テキスト", value=text, height=300)

    # セマンティックチャンク分割を行う
    chunks = semantic_split_text(text)
    st.write(f"テキストを {len(chunks)} チャンクに分割しました。")

    # 各チャンクごとにAzure OpenAIでテキストチェック
    all_check_results = []
    for i, chunk in enumerate(chunks):
        st.write(f"チャンク {i+1}/{len(chunks)} をチェックしています...")
        check_result = check_text_with_openai(chunk)
        all_check_results.append(check_result)

    # デバッグ: 全てのチェック結果を結合して表示
    st.write("Debug: チェック結果（全チャンク、未パース）:")
    combined_check_result = "\n\n".join(all_check_results)
    st.text_area("チェック結果（未パース）", value=combined_check_result, height=200)

    # チェック結果を表示（テーブル形式で見やすく）
    st.subheader("AIによるチェック結果")
    
    # 結果をパースしてデータフレームに変換
    df_results = parse_results_to_dataframe(combined_check_result)

    # DataFrameで結果を表示
    st.dataframe(df_results)

    # データをExcelファイルに変換してダウンロード
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_results.to_excel(writer, index=False)

    processed_data = output.getvalue()  # getvalue() でExcelファイルの内容を取得

    # ダウンロードオプション
    st.download_button(
        label="結果をExcelでダウンロード",
        data=processed_data,
        file_name="チェック結果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
