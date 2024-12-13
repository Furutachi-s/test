import os
import fitz  # PyMuPDFとしてインポート
import difflib
import re
import unicodedata
from tkinter import Tk, filedialog

# テキスト前処理関数（改行を維持）
def preprocess_text(text):
    # Unicode正規化
    text = unicodedata.normalize('NFKC', text)
    # 不要なスペースを削除（ただし改行は維持）
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

# 方法A：extract_text_with_blocks
def extract_text_with_blocks(pdf_document):
    page_texts = []
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # ブロック情報を取得
        blocks = page.get_text("dict")["blocks"]
        text_content = ""
        for block in blocks:
            if block["type"] == 0:  # テキストブロックの場合
                # ライン情報を取得してテキストを結合
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    text_content += line_text + "\n"  # ラインの終わりで改行
                text_content += "\n"  # ブロックの終わりで改行
        # テキストの前処理を適用
        text_content = preprocess_text(text_content)
        page_texts.append((page_num + 1, text_content))
    return page_texts

# 方法B：extract_text_with_pymupdf4llm
def extract_text_with_pymupdf4llm(pdf_document):
    from pymupdf4llm import to_markdown  # 必要に応じてインポート
    page_texts = []
    for page_num in range(len(pdf_document)):
        # ドキュメント全体を渡し、pagesパラメータでページを指定
        markdown_text = to_markdown(pdf_document, pages=[page_num])
        # テキストの前処理を適用
        markdown_text = preprocess_text(markdown_text)
        page_texts.append((page_num + 1, markdown_text))
    return page_texts

# テキストをファイルに保存する関数
def save_page_texts(page_texts, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for page_num, text in page_texts:
        with open(f"{folder_name}/page_{page_num}.txt", "w", encoding="utf-8") as f:
            f.write(text)

# テキストを比較して差分を表示する関数
def compare_texts(texts_a, texts_b):
    for (page_num_a, text_a), (page_num_b, text_b) in zip(texts_a, texts_b):
        if page_num_a != page_num_b:
            print(f"ページ番号が一致しません: 方法Aのページ{page_num_a}, 方法Bのページ{page_num_b}")
            continue
        print(f"\n===== ページ {page_num_a} の比較 =====")
        diff = difflib.unified_diff(
            text_a.splitlines(),
            text_b.splitlines(),
            fromfile='方法A',
            tofile='方法B',
            lineterm=''
        )
        diff_output = '\n'.join(diff)
        if diff_output:
            print(diff_output)
        else:
            print("差分はありません。")

# PDFファイルを選択する関数
def select_pdf_file():
    root = Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    return pdf_path

# メイン処理
def main():
    # PDFファイルを選択
    pdf_path = select_pdf_file()
    if not pdf_path:
        print("PDFファイルが選択されませんでした。")
        return

    # PDFを開く
    doc = fitz.open(pdf_path)

    # 方法Aでテキストを抽出
    texts_a = extract_text_with_blocks(doc)
    save_page_texts(texts_a, "method_a_texts")

    # 方法Bでテキストを抽出
    texts_b = extract_text_with_pymupdf4llm(doc)
    save_page_texts(texts_b, "method_b_texts")

    # テキストを比較
    compare_texts(texts_a, texts_b)

if __name__ == "__main__":
    main()
