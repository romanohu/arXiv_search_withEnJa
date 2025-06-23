import datetime
import numpy as np
import feedparser
import wikipediaapi
from janome.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer, util

# モデル・形態素解析器の準備
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
tokenizer = Tokenizer()


# Wikipediaから関連語を取得する関数
def get_wikipedia_related_words(query, lang='ja', top_k=20):
    wiki_wiki = wikipediaapi.Wikipedia(language=lang, user_agent='arxiv-search-app/1.0')

    # クエリが複数単語の場合は、各単語でもWikipediaページを探索
    # まず元のクエリで検索
    pages = []
    page = wiki_wiki.page(query)
    if page.exists():
        pages.append(page)
    else:
        # Janomeで分割して各単語でページを探索
        tokenizer = Tokenizer()
        tokens = [token.base_form for token in tokenizer.tokenize(query)
                  if token.part_of_speech.split(',')[0] in ['名詞', '動詞', '形容詞']
                  and token.base_form != '*']
        for token in tokens:
            p = wiki_wiki.page(token)
            if p.exists():
                pages.append(p)

    if not pages:
        return []

    # ページ内リンク（すべての内部リンク）を取得
    normal_links = []
    for p in pages:
        normal_links.extend(list(p.links.keys()))

    # 「関連項目」セクションに限定して単語を取得する再帰関数
    def get_related_section_text(section, target_title="関連項目"):
        if section.title == target_title:
            return section.text
        for subsection in section.sections:
            result = get_related_section_text(subsection, target_title)
            if result:
                return result
        return None

    # 「関連項目」セクションのテキストを抽出
    related_texts = []
    for p in pages:
        related_text = get_related_section_text(p)
        if related_text:
            related_texts.append(related_text)
    print(f"Related section texts: {related_texts}")

    # 関連語をテキストから抽出（改行を空白にし、単語分割）
    related_words = set()
    for related_text in related_texts:
        related_words.update(related_text.replace('\n', ' ').split())
    print(f"Related words extracted: {related_words}")

    # 通常リンクと関連項目語を統合し重複排除
    all_links = list(dict.fromkeys(normal_links + list(related_words)))

    return all_links


# 日本語文からキーワードを抽出する関数
def extract_keywords(text):
    keywords = []
    seen = set()
    # 形態素解析で名詞・動詞・形容詞を抽出
    for token in tokenizer.tokenize(text):
        base = token.base_form
        part = token.part_of_speech.split(',')[0]
        # 品詞と重複チェック、元の入力は除く
        if part in ['名詞', '動詞', '形容詞'] and base != '*' and base not in seen and base != text:
            keywords.append(base)
            seen.add(base)
    return keywords


# 関連語リストからベクトル類似度に基づいて上位を抽出
def get_dynamic_related_words(base_word, all_words, top_k=20):
    # 入力単語のベクトル
    base_vec = model.encode(base_word, convert_to_tensor=True)
    # 候補語全体をベクトル化
    all_vecs = model.encode(all_words, convert_to_tensor=True)
    # コサイン類似度を計算
    scores = util.cos_sim(base_vec, all_vecs)[0].cpu().numpy()
    # 類似度の高い順に並べて上位を返す
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [all_words[i] for i in sorted_indices]


# arXiv APIから論文を取得する関数
def fetch_arxiv_papers(query="transformer", max_results=1000, date_from=None, date_to=None):
    import urllib.parse
    base_url = "http://export.arxiv.org/api/query?"

    # 日付範囲をクエリに追加
    date_query = ""
    if date_from or date_to:
        from_str = date_from.strftime("%Y%m%d") if date_from else "00000000"
        to_str = date_to.strftime("%Y%m%d") if date_to else "99999999"
        date_query = f"+AND+submittedDate:[{from_str}+TO+{to_str}]"

    # クエリをURL形式に変換
    query_encoded = urllib.parse.quote(query)
    url = f"{base_url}search_query=all:{query_encoded}{date_query}&start=0&max_results={max_results}&sortBy=submittedDate"
    print(f"Fetching papers from: {url}")

    # RSSフィードから取得
    feed = feedparser.parse(url)
    papers = []

    # フィードのエントリを解析
    for entry in feed.entries:
        pub_date = datetime.datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").date()
        papers.append({
            "title": entry.title.strip(),
            "summary": entry.summary.strip().replace('\n', ' '),
            "authors": [a.name for a in entry.authors],
            "link": entry.link,
            "published": pub_date.strftime("%Y-%m-%d")
        })
    # 取得した論文の数を表示
    print(f"Fetched {len(papers)} papers from arXiv.")

    return papers


# 論文をタイトル＋要約でエンコードする関数
def encode_papers(papers):
    # タイトルと要約を結合してベクトル化
    texts = [p["title"] + ". " + p["summary"] for p in papers]
    return model.encode(texts, convert_to_numpy=True)


# モデルインスタンスを取得する関数
def get_model():
    return model
