import os
import streamlit as st
import pandas as pd
import json
import time
import re
import unicodedata
import asyncio
from openai import AzureOpenAI
from io import BytesIO

# Azureキー・エンドポイントを環境変数から取得
key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not key or not endpoint:
    st.error("Azure OpenAIのAPIキーまたはエンドポイントが設定されていません。")
    st.stop()

# Azure OpenAI クライアント
client = AzureOpenAI(api_key=key, api_version="2023-12-01-preview", azure_endpoint=endpoint)

chat_history = []
chat_logs = []

def normalize_text(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize('NFKC', str(text))
    # 記号や余計なスペースを除去
    text = re.sub(r'\s+', ' ', text).strip()
    return text

async def async_compare_errors_with_gpt(client, ai_error, error_list_error, semaphore):
    """
    AIエラーと誤記リストエラーをGPTで比較し、同じ誤りか判定する非同期関数
    戻り値: True(同じ誤り), False(異なる)
    """
    
    # 事前正規化処理追加
    ai_error_normalized = dict(ai_error)
    ai_error_normalized["指摘箇所"] = normalize_text(ai_error_normalized["指摘箇所"])
    ai_error_normalized["周辺テキスト"] = normalize_text(ai_error_normalized["周辺テキスト"])

    error_list_error_normalized = dict(error_list_error)
    error_list_error_normalized["誤記内容"] = normalize_text(error_list_error_normalized["誤記内容"])
    error_list_error_normalized["正しい内容"] = normalize_text(error_list_error_normalized["正しい内容"])

    async with semaphore:
        await asyncio.sleep(0.2) # レート回避用少待ち
        prompt = f'''
以下の2つのエラー情報が同じ誤りを指摘しているか判断してください。

判断基準の目安:
- エラータイプ(誤字、文法、文脈)が同様または非常に類似しているか
- 指摘箇所が同一ページで位置が近く、類似したテキストを対象としている
- 誤記内容とAI指摘内容が1~2文字程度の差異や表記揺れであれば同じエラーとみなす
- 確信が持てなくても、かなり近い場合は「はい」と答えてください

回答は「はい」または「いいえ」のみ。

---
AIの指摘：
- ページ番号：{ai_error_normalized["ページ番号"]}
- カテゴリ：{ai_error_normalized["カテゴリ"]}
- 指摘箇所：{ai_error_normalized["指摘箇所"]}
- 指摘理由：{ai_error_normalized["指摘理由"]}
- 周辺テキスト：{ai_error_normalized["周辺テキスト"]}

誤記リストのエラー：
- ページ番号：{error_list_error_normalized["ページ"]}
- 誤記内容：{error_list_error_normalized["誤記内容"]}
- 正しい内容：{error_list_error_normalized["正しい内容"]}

これらは同じエラーとみなせますか？「はい」または「いいえ」で回答してください。
'''
        session_message = [{"role": "user", "content": prompt}]
        chat_logs.append({"comparison": {"ai_error": ai_error_normalized, "error_list_error": error_list_error_normalized, "prompt": prompt}})

        for attempt in range(3):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o",
                    seed=42,
                    temperature=0,
                    max_tokens=2,
                    messages=session_message
                )
                answer = response.choices[0].message.content.strip()
                chat_history.append({
                    "comparison": {
                        "ai_error": ai_error_normalized,
                        "error_list_error": error_list_error_normalized,
                        "prompt": prompt,
                        "response": answer
                    }
                })
                chat_logs[-1]["comparison"]["response"] = answer
                return (ai_error, error_list_error, (answer == "はい"))
            except Exception as ex:
                await asyncio.sleep(5)

        chat_history.append({
            "comparison": {
                "ai_error": ai_error_normalized,
                "error_list_error": error_list_error_normalized,
                "prompt": prompt,
                "response": "エラー"
            }
        })
        chat_logs[-1]["comparison"]["response"] = "エラー"
        return (ai_error, error_list_error, False)

async def run_evaluation_gather(client, filtered_results_df, error_df, progress_bar, status_text):
    filtered_results_df['ページ番号'] = filtered_results_df['ページ番号'].astype(int)
    error_df['ページ'] = error_df['ページ'].astype(int)

    pairs = []
    pages = set(filtered_results_df['ページ番号']).union(set(error_df['ページ']))
    for page_num in pages:
        ai_errors_on_page = filtered_results_df[filtered_results_df['ページ番号'] == page_num]
        error_list_errors_on_page = error_df[error_df['ページ'] == page_num]
        for idx_ai, ai_error in ai_errors_on_page.iterrows():
            for idx_el, el_error in error_list_errors_on_page.iterrows():
                pairs.append((idx_ai, idx_el, ai_error.to_dict(), el_error.to_dict()))

    if not pairs:
        return (0,0,0,0, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    semaphore = asyncio.Semaphore(5)
    tasks = [async_compare_errors_with_gpt(client, ai_err, el_err, semaphore) for (idx_ai, idx_el, ai_err, el_err) in pairs]
    results = await asyncio.gather(*tasks) # 順序保証

    # results[i]は (ai_error_dict, error_list_error_dict, matched_bool)
    # pairs[i]対応：pairs[i][0]=idx_ai, pairs[i][1]=idx_el
    matched_ai_errors = set()
    matched_error_list_errors = set()
    for i, r in enumerate(results):
        ai_err, el_err, matched = r
        idx_ai, idx_el = pairs[i][0], pairs[i][1]
        if matched and (idx_ai not in matched_ai_errors) and (idx_el not in matched_error_list_errors):
            matched_ai_errors.add(idx_ai)
            matched_error_list_errors.add(idx_el)

    TP = len(matched_ai_errors)
    # FP: マッチしなかったAIエラー
    FP = len(filtered_results_df) - TP
    # FN: 誤記リストでマッチしなかったもの
    FN = len(error_df) - len(matched_error_list_errors)
    TN = 0  # 必要なら別途計算

    matched_ai_df = filtered_results_df.loc[list(matched_ai_errors)] if matched_ai_errors else pd.DataFrame()
    matched_error_list_df = error_df.loc[list(matched_error_list_errors)] if matched_error_list_errors else pd.DataFrame()
    false_positives = filtered_results_df.loc[[i for i in filtered_results_df.index if i not in matched_ai_errors]]
    false_negatives = error_df.loc[[i for i in error_df.index if i not in matched_error_list_errors]]

    return (TP, FP, FN, TN, matched_ai_df, matched_error_list_df, false_positives, false_negatives)

# Streamlit UI
st.title("精度評価ツール")

filtered_file = st.file_uploader("フィルタ後のAIチェック結果ファイルをアップロードしてください (Excel)", type="xlsx")
error_list_file = st.file_uploader("誤記リストファイルをアップロードしてください (CSVまたはExcel)", type=["csv","xlsx"])

if filtered_file is not None and error_list_file is not None:
    # 読み込み
    filtered_results_df = pd.read_excel(filtered_file)
    # 必須列チェック(簡易)
    if not set(["ページ番号","カテゴリ","指摘箇所","指摘理由","周辺テキスト"]).issubset(filtered_results_df.columns):
        st.error("フィルタ後チェック結果ファイルの形式が不正です。必須列がありません。")
    else:
        # 誤記リスト読み込み
        if error_list_file.name.endswith(".csv"):
            try:
                error_df = pd.read_csv(error_list_file, encoding='utf-8')
            except UnicodeDecodeError:
                error_df = pd.read_csv(error_list_file, encoding='shift_jis')
        else:
            error_df = pd.read_excel(error_list_file)

        if not set(["ページ","誤記内容","正しい内容"]).issubset(error_df.columns):
            st.error("誤記リストファイルに必須列(ページ,誤記内容,正しい内容)がありません。")
        else:
            if st.button("精度評価開始"):
                st.subheader("精度評価進捗状況")
                evaluation_progress_bar = st.progress(0)
                evaluation_status_text = st.empty()

                start_time = time.time()
                TP, FP, FN, TN, matched_ai_df, matched_error_list_df, false_positives, false_negatives = asyncio.run(
                    run_evaluation_gather(client, filtered_results_df, error_df, evaluation_progress_bar, evaluation_status_text)
                )

                evaluation_status_text.write("精度評価が完了しました。")

                st.subheader("精度評価結果")
                st.write(f"TP: {TP}")
                st.write(f"FP: {FP}")
                st.write(f"FN: {FN}")
                st.write(f"TN: {TN}")

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")
                st.write(f"F1 Score: {f1_score:.2f}")

                end_time = time.time()
                total_time = end_time - start_time
                st.write(f"処理時間: {total_time:.2f}秒")

                # 結果Excel出力
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    filtered_results_df.to_excel(writer, sheet_name='フィルタ後チェック結果', index=False)
                    matched_ai_df.to_excel(writer, sheet_name='マッチしたAIエラー', index=False)
                    matched_error_list_df.to_excel(writer, sheet_name='マッチした誤記リスト', index=False)
                    false_positives.to_excel(writer, sheet_name='False Positives', index=False)
                    false_negatives.to_excel(writer, sheet_name='False Negatives', index=False)
                    summary_df = pd.DataFrame({
                        'TP':[TP],'FP':[FP],'FN':[FN],'TN':[TN],
                        'Precision':[precision],'Recall':[recall],'F1':[f1_score],
                        '処理時間（秒）':[total_time]
                    })
                    summary_df.to_excel(writer, sheet_name='精度評価結果', index=False)
                    chat_logs_df = pd.DataFrame(chat_logs)
                    chat_logs_df.to_excel(writer, sheet_name='チャットログ', index=False)
                output.seek(0)

                st.download_button(
                    label="精度評価結果をExcelでダウンロード",
                    data=output.getvalue(),
                    file_name="精度評価結果.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

else:
    st.write("フィルタ後結果ファイルと誤記リストファイルをアップロードしてください。")
