import os
import json
import pandas as pd
import time
from openai import AzureOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from PyPDF2 import PdfReader # PDFファイルを読み込むためのライブラリ
# インポート部分の変更
# from PyPDF2 import PdfFileReader を from PyPDF2 import PdfReader に変更。

######################################################################################
#変数設定部
######################################################################################
db_name = "db_emb" # ベクターストアの名前
db_directory = "database-pdf" # ベクターストアの保存先のディレクトリ(1-pdfファイル参照ボット向)

batch_size = 100 #普段は特に調整しない
######################################################################################
#関数定義部
######################################################################################
def get_azure_config(path):
    """ChatGPTのAPI情報とEmbeddingのAPI情報を取得する"""
    with open(path, "r", encoding="UTF-8") as f:
        azure_config = json.load(f)

    # ChatGPT4のAPI情報取得
    gpt4_id = azure_config["gpt4"]["MODEL_ID"]
    client = AzureOpenAI(
    api_key = azure_config["gpt4"]["API_KEY"],  
    api_version = azure_config["gpt4"]["API_VERSION"],
    azure_endpoint = azure_config["gpt4"]["API_BASE"]
    )
    # EmbeddingのAPI情報取得(langchain用、環境変数に設定する)
    emb_id = azure_config["embed"]["MODEL_ID"]
    os.environ["AZURE_OPENAI_API_KEY"] = azure_config["embed"]["API_KEY"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_config["embed"]["API_BASE"]
    os.environ["OPENAI_API_VERSION"] = azure_config["embed"]["API_VERSION"]
    
    return client, gpt4_id, emb_id

######################################################################################
#初期化処理
######################################################################################

print("""
 _  _____  __  __    _  _____ ____  _   _
| |/ / _ \|  \/  |  / \|_   _/ ___|| | | |
| ' / | | | |\/| | / _ \ | | \___ \| | | |
| . \ |_| | |  | |/ ___ \| |  ___) | |_| |
|_|\_\___/|_|  |_/_/   \_\_| |____/ \___/
""")
# Azureの設定取得
print("loading azure config...")
client, model_id, emb_id = get_azure_config("config/azure_openai.json")

# LangchainのEmbedding設定
embeddings = AzureOpenAIEmbeddings(azure_deployment=emb_id)

# FAISSの初期化
print("check embedding dimension...")
test_emb = embeddings.embed_documents(["Hello World!"])
dim = len(test_emb[0])
print(f"embedding dimension: {dim}")

db_faiss = FAISS(
    embedding_function = embeddings,
    index=IndexFlatL2(dim),
    docstore=InMemoryDocstore(),
    index_to_docstore_id = {}
)
print("reading dataset...")
######################################################################################
#データ読み込み部
#データによってここの処理を変える、基本的にはDocument形式のデータにしてdoc_listに追加する
#メタデータのfileNameはRAG使用時に参考文書名として使用する
######################################################################################
# PDFファイルを読み込む

pdf_path = './example/AI Act_Regulation 2024-1689_EUOJ_12072024.pdf' # PDFファイルのパス
pdf = PdfReader(open(pdf_path, 'rb')) # PDFファイルを読み込む
# インポート部分の変更
# PdfFileReader を PdfReader に変更。

doc_list = []
# for page_num in range(pdf.getNumPages()): # ページ数分繰り返す
for page_num in range(len(pdf.pages)):      # ページ数分繰り返す
#    page = pdf.getPage[page_num]           # ページを取得
    page = pdf.pages[page_num]              # ページを取得
    content = page.extract_text()           # ページのテキストを取得
    metadata = {"fileName": f"Page {page_num+1}"} # メタデータの設定
    doc = [content, metadata]
    doc_list.append(doc)

######################################################################################
#ベクトル化とベクターストア保存
######################################################################################
print("embedding dataset...")
def chunks(lst, n):
    """リストをn個ずつに分割"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

faiss_content_list = []
faiss_metadata_list = []
embedding_num_retry = 5
embedding_retry_interval = 5

for chunk in tqdm(chunks(doc_list, batch_size), total=len(doc_list)//batch_size): # バッチサイズ(100)ごとに処理
    content_list = [c[0] for c in chunk]
    metadata_list = [c[1] for c in chunk]

    for i in range(embedding_num_retry):
        try:
            emb = embeddings.embed_documents(content_list)
            break
        except Exception as e:
            print(f"embedding error: {e}")
            print("retrying...")
            time.sleep(embedding_retry_interval)

    for c,e in zip(content_list, emb):
        faiss_content_list.append([c,e])
    faiss_metadata_list.extend(metadata_list)

db_faiss.add_embeddings(faiss_content_list, faiss_metadata_list)

if not os.path.exists(db_directory):
    os.makedirs(db_directory)
persist_directory = os.path.join(db_directory, db_name) 
# ローカルに保存する
db_faiss.save_local(persist_directory)
