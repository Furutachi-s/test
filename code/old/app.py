import os
import streamlit as st
import fitz  # PyMuPDFとしてインポート
from openai import AzureOpenAI
import pandas as pd
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 追加

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

# チャンク分割の関数を追加
def split_text_into_chunks(text):
    # Debug: テキストの分割を開始
    st.write("テキストの分割を開始します...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 各チャンクの最大文字数
        chunk_overlap=100  # チャンク同士のオーバーラップ部分の長さ
    )
    chunks = splitter.split_text(text)
    
    # Debug: 分割されたチャンクの数を表示
    st.write(f"テキストは {len(chunks)} チャンクに分割されました")
    
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

    # チャンク分割を行う
    chunks = split_text_into_chunks(text)
    
    # 各チャンクごとにAzure OpenAIでテキストチェック
    all_check_results = []
    for i, chunk in enumerate(chunks):
        # Debug: 各チャンクの内容を確認
        st.write(f"チャンク {i+1}/{len(chunks)} の内容: {chunk[:100]}...")  # チャンクの最初の100文字のみ表示
        st.write(f"チャンク {i+1}/{len(chunks)} をチェックしています...")
        
        # チャンクが空でないことを確認
        if chunk.strip():
            check_result = check_text_with_openai(chunk)
            all_check_results.append(check_result)
        else:
            st.write(f"チャンク {i+1} は空なのでスキップされました。")

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
