# utils.py
import datetime
import numpy as np
import feedparser
import wikipediaapi
from janome.tokenizer import Tokenizer
from sentence_transformers import SentenceTransformer, util

# モデル・形態素解析器の準備
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
tokenizer = Tokenizer()

def get_wikipedia_related_words(query, lang='ja', top_k=10):
    wiki_wiki = wikipediaapi.Wikipedia(language=lang, user_agent='arxiv-search-app/1.0')
    page = wiki_wiki.page(query)
    if not page.exists():
        return []

    # 通常のリンク一覧（ページ全体から）
    normal_links = list(page.links.keys())

    # 「関連項目」セクション内のテキストを再帰的に探す
    def get_related_section_text(section, target_title="関連項目"):
        if section.title == target_title:
            return section.text
        for subsection in section.sections:
            result = get_related_section_text(subsection, target_title)
            if result:
                return result
        return None

    related_text = get_related_section_text(page)
    print(f"Related section text: {related_text}")


    if related_text:
        related_words = set(related_text.replace('\n', ' ').split())
        print(f"Related words extracted: {related_words}")
        all_links = list(dict.fromkeys(normal_links + list(related_words)))
    else:
        all_links = normal_links

    return all_links

def extract_keywords(text):
    keywords = []
    seen = set()
    for token in tokenizer.tokenize(text):
        base = token.base_form
        part = token.part_of_speech.split(',')[0]
        # 入力textと同じ語も追加しない
        if part in ['名詞', '動詞', '形容詞'] and base != '*' and base not in seen and base != text:
            keywords.append(base)
            seen.add(base)
    return keywords

def get_dynamic_related_words(base_word, all_words, top_k=20):
    base_vec = model.encode(base_word, convert_to_tensor=True)
    all_vecs = model.encode(all_words, convert_to_tensor=True)
    scores = util.cos_sim(base_vec, all_vecs)[0].cpu().numpy()
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [all_words[i] for i in sorted_indices]

def fetch_arxiv_papers(query="transformer", max_results=500, sort_order="descending", date_from=None, date_to=None):
    base_url = "http://export.arxiv.org/api/query?"
    query = query.replace(" ", "+")
    url = f"{base_url}search_query=all:{query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder={sort_order}"
    feed = feedparser.parse(url)
    papers = []
    for entry in feed.entries:
        pub_date = datetime.datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ").date()
        if date_from and pub_date < date_from:
            continue
        if date_to and pub_date > date_to:
            continue
        papers.append({
            "title": entry.title.strip(),
            "summary": entry.summary.strip().replace('\n', ' '),
            "authors": [a.name for a in entry.authors],
            "link": entry.link,
            "published": pub_date.strftime("%Y-%m-%d")
        })
    return papers

def encode_papers(papers):
    texts = [p["title"] + ". " + p["summary"] for p in papers]
    return model.encode(texts, convert_to_numpy=True)

def get_model():
    return model
