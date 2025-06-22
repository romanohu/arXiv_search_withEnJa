import streamlit as st
import datetime
import numpy as np
from sentence_transformers import util

from utils import (
    get_wikipedia_related_words,
    extract_keywords,
    get_dynamic_related_words,
    fetch_arxiv_papers,
    encode_papers,
    get_model
)

model = get_model()

st.title("arXiv論文検索")
with st.expander("使い方"):
    st.write("このアプリは、arXivから論文を検索し、関連する論文を見つけるためのものです。以下の手順で使用できます:" \
    "\n" \
    "1. **arXiv分野検索キーワード**: 検索したいarXivの分野を「英語」で入力します。例:Machine Learning, transformerなど。" \
    "\n" \
    "2. **日本語クエリ**: 検索したいキーワードやフレーズを日本語で入力します。" \
    "\n" \
    "3. **形態素解析の使用**: 日本語クエリを形態素解析で分割するかどうかを選択します。" \
    "\n" \
    "4. **取得開始日と終了日**: 論文の取得期間を指定します。" \
    "\n" \
    "5. **発行日の並び順**: 論文の発行日の並び順を選択します。" \
    "\n" \
    "6. **並び替えの基準**: 検索結果の並び替え基準を選択します。" \
    "\n" \
    "7. **関連語の選択**: Wikipediaや形態素解析から抽出された関連語の候補から、必要なものを選択します。" \
    "\n" \
    "8. **検索結果の表示**: 検索結果が表示されます。各論文のタイトル、著者、発行日、要約、類似度スコアが表示されます。" \
    "\n" \
    "9. **著者で絞り込み**: 著者名で絞り込むことができます。" \
    "\n" \
    "10. **要約の表示**: 各論文の要約を表示することができます。")   

arxiv_keyword = st.text_input("arXiv分野検索キーワード(英語)")
query = st.text_input("日本語クエリ")
use_morph = st.checkbox("日本語クエリを形態素解析で分割する", value=True)

col1, col2 = st.columns(2)
with col1:
    date_from = st.date_input("📅 取得開始日", value=datetime.date.today() - datetime.timedelta(days=30))
with col2:
    date_to = st.date_input("📅 取得終了日", value=datetime.date.today())

sort_order = st.radio("発行日の並び順を選択してください", options=["新しい順（降順）", "古い順（昇順）"], index=0)
user_wants_desc = "降順" in sort_order
sort_type = st.radio("並び替えの基準", options=["スコア順", "発行日順"], index=0)

if query and arxiv_keyword:
    if use_morph:
        keywords = extract_keywords(query)
        keywords.append(query)
    else:
        keywords = [query]
    st.write("抽出されたキーワード:", ", ".join(keywords))

    base = query
    candidates = get_wikipedia_related_words(base)
    if candidates:
        candidates = get_dynamic_related_words(base, candidates)

    selected = []
    if candidates:
        st.write("💡 関連語の候補を選んでください:")
        selected = [word for word in candidates if st.checkbox(word)]

    final_query = " ".join(keywords + selected)
    st.markdown(f"**拡張後クエリ:** `{final_query}`")
    query_vec = model.encode(final_query, convert_to_numpy=True)

    with st.spinner("arXivから論文取得中..."):
        papers = fetch_arxiv_papers(
            query=arxiv_keyword,
            max_results=500,
            sort_order="descending",
            date_from=date_from,
            date_to=date_to
        )

        if not papers:
            st.warning("指定された期間に該当する論文が見つかりませんでした。")
        else:
            st.success(f"✅ {len(papers)}件の論文を取得しました")
            for p in papers:
                p["published_dt"] = datetime.datetime.strptime(p["published"], "%Y-%m-%d")

            selected_authors = st.multiselect("👤 著者で絞り込み", sorted(set(a for p in papers for a in p["authors"])))
            if selected_authors:
                papers = [p for p in papers if any(a in selected_authors for a in p["authors"])]

            with st.spinner("🔄 スコア計算中..."):
                paper_vecs = encode_papers(papers)
                scores = util.cos_sim(query_vec, paper_vecs)[0].cpu().numpy()
                papers = [p | {"score": scores[i]} for i, p in enumerate(papers) if scores[i] >= 0.2]

            if sort_type == "スコア順":
                papers = sorted(papers, key=lambda x: x["score"], reverse=True)
            else:
                papers = sorted(papers, key=lambda x: x["published_dt"], reverse=user_wants_desc)

            results_per_page = 10
            total_results = len(papers)
            total_pages = (total_results - 1) // results_per_page + 1
            selected_page = st.number_input("📄 表示ページを選んでください", 1, total_pages, 1)
            start_idx = (selected_page - 1) * results_per_page
            end_idx = start_idx + results_per_page

            st.subheader(f"📄 検索結果（ページ {selected_page} / {total_pages}）")
            for idx, p in enumerate(papers[start_idx:end_idx], start=start_idx + 1):
                st.markdown(f"### {idx}. [{p['title']}]({p['link']})")
                st.write(f"**Authors**: {', '.join(p['authors'])}")
                st.write(f"**Published**: {p['published']}")
                st.write(f"**Similarity**: `{p['score']:.4f}`")
                st.progress(min(int(p['score'] * 100), 100))
                with st.expander("📖 要約を表示"):
                    st.write(p["summary"])
