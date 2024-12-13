import fitz  # PyMuPDFとしてインポート
import re
import unicodedata
from tkinter import Tk, filedialog

# テキスト前処理関数
def preprocess_text(text):
    # Unicode正規化
    text = unicodedata.normalize('NFKC', text)
    # 不要なスペースを削除（ただし改行は維持）
    text = re.sub(r'[ \t]+', ' ', text)  # スペースとタブを一つのスペースに置換
    return text.strip()

# PDFからテキストをブロック単位で抽出する関数
def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    page_texts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # ブロック単位でテキスト抽出
        blocks = page.get_text("blocks")
        page_blocks = []
        for block in blocks:
            block_text = preprocess_text(block[4])  # ブロック内のテキストを前処理
            page_blocks.append(block_text)
        page_texts.append((page_num + 1, page_blocks))
    return page_texts

# 総文数をカウントする関数
def count_total_sentences(page_texts):
    total_text = ''
    for page_num, blocks in page_texts:
        total_text += '\n'.join(blocks) + '\n'
    # 日本語の文末句読点で分割
    sentences = re.split(r'[。．！？]', total_text)
    # 空の文字列を除外
    sentences = [s for s in sentences if s.strip()]
    total_sentences = len(sentences)
    return total_sentences

# PDFファイルを選択する関数
def select_pdf_file():
    # Tkinterのインスタンスを非表示で生成
    root = Tk()
    root.withdraw()
    # ファイル選択ダイアログを表示
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    return pdf_path

# PDFファイルのパスを取得
pdf_path = select_pdf_file()

if pdf_path:  # ファイルが選択された場合のみ実行
    # テキスト抽出
    page_texts = extract_text_blocks(pdf_path)

    # 抽出されたテキストを表示（ページごと・ブロックごとに表示）
    for page_num, blocks in page_texts:
        print(f"ページ {page_num}:")
        for block in blocks:
            print(block)
            print("-" * 20)
        print("=" * 40)

    # 総文数のカウント
    total_sentences = count_total_sentences(page_texts)
    print(f"総文数: {total_sentences}")
else:
    print("PDFファイルが選択されませんでした。")
