import streamlit as st
import datetime
import numpy as np
from sentence_transformers import util

# 外部ユーティリティ関数をインポート
from utils import (
    get_wikipedia_related_words,
    extract_keywords,
    get_dynamic_related_words,
    fetch_arxiv_papers,
    encode_papers,
    get_model
)

# 埋め込みモデルの読み込み
model = get_model()

# タイトルと説明
st.title("arXiv論文検索")
with st.expander("使い方"):
    st.write("これは英語でしか検索できないarXivに対して日本語での検索を試みる検索システムです．以下の手順で使用できます:" \
    "\n" \
    "1. **arXiv分野検索キーワード**: arXivで調べたい論文が該当する大まかな分野を「英語」で入力します．例:Machine Learning, LLMなど．" \
    "\n" \
    "2. **取得開始日と終了日**: 論文の取得期間を指定します．" \
    "\n" \
    "3. **日本語クエリ**: より詳細な検索したいキーワードやフレーズを「日本語」で入力します．" \
    "\n" \
    "4.   **形態素解析の使用**: 日本語クエリを形態素解析で分割するかどうかを選択します．" \
    "\n" \
    "5. **wikipediaの言語選択**: 関連語を提示するために使用するwikipediaの言語を指定できます" \
    "\n" \
    "6. **発行日の並び順**: 論文の発行日の並び順を選択します．" \
    "\n" \
    "7. **並び替えの基準**: 検索結果の並び替え基準を選択します．" \
    "\n" \
    "8. **関連語の選択**: Wikipediaや形態素解析から抽出された関連語の候補から，必要なものを選択します．" \
    "\n" \
    "9. **検索結果の表示**: 検索結果が表示されます．各論文のタイトル，著者，発行日，要約，類似度スコアが表示されます．" \
    "\n" \
    "10. **著者で絞り込み**: 著者名で絞り込むことができます．" \
    "\n" \
    "11. **要約の表示**: 各論文の要約を表示することができます．")   

# ユーザ入力: arXiv分野
arxiv_keyword = st.text_input("arXiv分野検索キーワード(英語)")

# ユーザ入力: 期間選択
col1, col2 = st.columns(2)
with col1:
    date_from = st.date_input("📅 取得開始日", value=datetime.date.today() - datetime.timedelta(days=30))
with col2:
    date_to = st.date_input("📅 取得終了日", value=datetime.date.today())

# 日本語クエリと形態素解析オプション
query = st.text_input("日本語クエリ")
use_morph = st.checkbox("日本語クエリを形態素解析で分割する", value=True)

# Wikipedia使用言語の選択
wiki_lang = st.radio("Wikipediaの言語", options=["日本語", "English"], index=0)
wiki_lang_code = "ja" if wiki_lang == "日本語" else "en"

# 並び替え設定
sort_type = st.radio("並び替えの基準", options=["スコア順", "発行日順"], index=0)
sort_order = st.radio("並び順を選択してください", options=["新しい順（降順）", "古い順（昇順）"], index=0)

# 並び替えに関する設定変数を決定
if sort_type == "スコア順":
    reverse_sort = "降順" in sort_order
    sort_key = "score"
else:
    reverse_sort = "降順" in sort_order
    sort_key = "published_dt"

# クエリとキーワードが入力されたときに処理を開始
if query and arxiv_keyword:
    # 形態素解析の使用有無によってクエリを処理
    if use_morph:
        keywords = extract_keywords(query)
        keywords.append(query)  # 元の語も加える
    else:
        keywords = [query]

    # 抽出キーワード表示
    st.write("抽出されたキーワード:", ", ".join(keywords))

    base = query

    # Wikipediaから関連語を取得
    candidates = get_wikipedia_related_words(base, lang=wiki_lang_code)
    if candidates:
        candidates = get_dynamic_related_words(base, candidates)

    # ユーザによる関連語の選択
    selected = []
    if candidates:
        st.write("💡 関連語の候補を選んでください:")
        cols = st.columns(2)
        for i, word in enumerate(candidates):
            if cols[i % 2].checkbox(word):
                selected.append(word)

    # 拡張後クエリの作成と表示
    final_query = " ".join(keywords + selected)
    st.markdown(f"**拡張後クエリ:** `{final_query}`")

    # クエリベクトルをエンコード
    query_vec = model.encode(final_query, convert_to_numpy=True)

    # 論文の取得処理開始
    with st.spinner("arXivから論文取得中..."):
        papers = fetch_arxiv_papers(
            query=arxiv_keyword,
            max_results=1000,
            date_from=date_from,
            date_to=date_to
        )

        # 該当する論文がない場合の処理
        if not papers:
            st.warning("指定された期間に該当する論文が見つかりませんでした．")
        else:
            st.success(f"✅ {len(papers)}件の論文を取得しました")

            # 発行日をdatetime型に変換
            for p in papers:
                p["published_dt"] = datetime.datetime.strptime(p["published"], "%Y-%m-%d")

            # 著者で絞り込み
            selected_authors = st.multiselect("👤 著者で絞り込み", sorted(set(a for p in papers for a in p["authors"])))
            if selected_authors:
                papers = [p for p in papers if any(a in selected_authors for a in p["authors"])]

            # 類似度スコア計算
            with st.spinner("🔄 スコア計算中..."):
                paper_vecs = encode_papers(papers)
                scores = util.cos_sim(query_vec, paper_vecs)[0].cpu().numpy()
                papers = [p | {"score": scores[i]} for i, p in enumerate(papers)]

            # 並び替えの実行
            if sort_key == "score":
                papers = sorted(papers, key=lambda x: x["score"], reverse=reverse_sort)
            else:
                papers = sorted(papers, key=lambda x: x["published_dt"], reverse=reverse_sort)

            # ページネーション
            results_per_page = 10
            total_results = len(papers)
            total_pages = (total_results - 1) // results_per_page + 1
            selected_page = st.number_input("📄 表示ページを選んでください", 1, total_pages, 1)
            start_idx = (selected_page - 1) * results_per_page
            end_idx = start_idx + results_per_page

            # 検索結果の表示
            st.subheader(f"📄 検索結果（ページ {selected_page} / {total_pages}）")
            for idx, p in enumerate(papers[start_idx:end_idx], start=start_idx + 1):
                st.write(f"{idx}. [{p['title']}]({p['link']})")
                st.write(f"**Authors**: {', '.join(p['authors'])}")
                st.write(f"**Published**: {p['published']}")
                st.write(f"**Similarity**: `{p['score']:.4f}`")
                # 類似度スコアに応じた進捗バー
                progress_value = min(max(int(p['score'] * 100), 0), 100)
                st.progress(progress_value)
                with st.expander("📖 要約を表示"):
                    st.write(p["summary"])
