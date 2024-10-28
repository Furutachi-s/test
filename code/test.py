import os
from azure.identity import DefaultAzureCredential
from azure.ai.openai import OpenAIClient

# 認証情報の設定
api_key = os.getenv("AZURE_OPENAI_KEY")
if api_key is None:
    raise ValueError("API キーが設定されていません。環境変数 'AZURE_OPENAI_KEY' を設定してください。")

endpoint = "https://aoai-furutachi-us.openai.azure.com/"

# クライアントの初期化
client = OpenAIClient(
    endpoint=endpoint,
    credential=api_key
)

# Embeddings のテスト
try:
    response = client.get_embeddings(
        deployment_id="text-embedding-3-small",
        input="テスト"
    )
    embeddings = response.data[0].embedding
    print("Embeddings のテスト成功:", embeddings[:5], "...")
except Exception as e:
    print("Embeddings のテスト失敗:", e)

# Chat モデルのテスト
try:
    response = client.get_chat_completions(
        deployment_id="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "こんにちは、元気ですか？"}
        ]
    )
    print("Chat モデルのテスト成功:", response.choices[0].message.content)
except Exception as e:
    print("Chat モデルのテスト失敗:", e)
