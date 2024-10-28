import pymupdf4llm
import fitz
import boto3
import json

def initialize_app():
    client = ChatClient()
    session_message = []
    return client, session_message
    
class ChatClient:
    def __init__(self):
        self.client = boto3.client('bedrock-runtime', region_name='us-east-1')
        self.model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'

    def create_minutes(self, session_messages):
        print("Creating minutes...")
        new_session_message = []
        for message in session_messages:
            new_session_message.append(message)
        message = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5000,
                "system": """
                    入力された情報は、建設機械の取扱説明書をテキストに文字起こしをしたものです。
                    建設機械で使用されている部品名や、漢字の誤字、文法的におかしな表現を摘出してください。
                    注意事項:pdfをテキストに変換する際、不要なスペースが含まれることがあるので、抽出した内容の中で、不要なスペースが含まれることに関連する指摘は削除して、それ以外の修正必要個所のみ抽出してください。
                """,               
              "messages": new_session_message  # 展開してリストに追加
            })
        )
        response_body = json.loads(message.get('body').read())
        return response_body['content'][0]['text']

def main():
    pdf_path = "./WA470_10_SM_Sec40.pdf"
    # PDFドキュメントを開く
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    client, session_message = initialize_app()

    for page_num in range(total_pages):
        md_text = pymupdf4llm.to_markdown(pdf_path, pages=[page_num])
        session_message.append({"role": "user", "content": md_text})
        summary = client.create_minutes(session_message)
        with open(f"minutes_page_{page_num + 1}.txt", 'w', newline='', encoding='utf-8') as file:
            file.write(summary)
        session_message = []

if __name__ == "__main__":
    main()