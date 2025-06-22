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
    wiki_wiki = wikipediaapi.Wikipedia(language=lang, user_agent='arxiv-search-app/1.0 (yourname@example.com)')
    page = wiki_wiki.page(query)
    if not page.exists():
        return []
    links = list(page.links.keys())
    return links[:top_k]

def extract_keywords(text):
    keywords = []
    for token in tokenizer.tokenize(text):
        base = token.base_form
        part = token.part_of_speech.split(',')[0]
        if part in ['名詞', '動詞', '形容詞'] and base != '*':
            keywords.append(base)
    return keywords

def get_dynamic_related_words(base_word, all_words, top_k=5):
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
