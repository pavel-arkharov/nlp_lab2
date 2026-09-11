#!/usr/bin/env python3
"""NLP Laboratory 2: document retrieval and semantic similarity."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import certifi
import nltk
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk import pos_tag, word_tokenize
from nltk.corpus import brown, stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent
NLTK_DATA = ROOT / "data" / "nltk_data"
SEED = 2026
WORD = re.compile(r"^[A-Za-z]+$")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PAIR = ("potato", "tiger")
REQUIRED_PAIR = ("house", "train")
URLS = (
    "https://www.bbc.com/news/articles/cdj4z2x3zyko",
    "https://www.bbc.com/news/articles/cpvelrmp1vmo",
    "https://www.bbc.com/news/articles/c62jnyn2n2do",
    "https://www.bbc.com/news/articles/cew9nkx7v9eo",
    "https://www.bbc.com/news/articles/c62jln7n2vvo",
)
PENN_TAGS = (
    "$", "''", "(", ")", ",", "--", ".", ":", "CC", "CD", "DT", "EX",
    "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS",
    "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM",
    "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP",
    "WP$", "WRB", "``",
)
FAMILY = {
    "house": "house.n.01", "apartment": "apartment.n.01",
    "flat": "apartment.n.01", "hotel": "hotel.n.01", "room": "room.n.01",
    "living room": "living_room.n.01", "toilet": "toilet.n.01",
}
NLTK_RESOURCES = ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                  "brown", "averaged_perceptron_tagger_eng")


def ensure_directories() -> None:
    (ROOT / "outputs").mkdir(exist_ok=True)
    NLTK_DATA.mkdir(parents=True, exist_ok=True)


def ensure_nltk_data() -> None:
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    if str(NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA))
    for package in NLTK_RESOURCES:
        nltk.download(package, download_dir=str(NLTK_DATA), quiet=True)


def extract_bbc_article(html: str, article_id: str, url: str) -> dict[str, Any]:
    """A1: Extract the article body and basic metadata with BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    if article is None:
        raise ValueError(f"{article_id}: BBC article element was not found")
    nodes = article.select('[data-component="text-block"] p') or article.find_all("p")
    text = "\n\n".join(node.get_text(" ", strip=True) for node in nodes)
    if len(text) < 500:
        raise ValueError(f"{article_id}: extracted article text is unexpectedly short")
    meta = lambda name: (soup.find("meta", attrs={"property": name}) or {}).get("content", "")
    return {
        "id": article_id, "url": url, "title": meta("og:title"),
        "image_url": meta("og:image"), "paragraph_count": len(nodes), "text": text,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def preprocess_tokens(tokens: Sequence[str]) -> list[str]:
    """A2: Filter, POS-tag with original casing, lowercase, and lemmatize."""
    blocked = set(stopwords.words("english"))
    tagged = pos_tag([t for t in tokens if WORD.fullmatch(t) and len(t) > 2])
    pos = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(token.lower(), pos.get(tag[0], wn.NOUN))
            for token, tag in tagged if token.lower() not in blocked]


def load_articles() -> list[dict[str, Any]]:
    """Download and prepare the five selected articles."""
    articles = []
    for number, url in enumerate(URLS, 1):
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        article = extract_bbc_article(response.text, f"D{number}", url)
        raw = word_tokenize(article.pop("text"))
        normalized = preprocess_tokens(raw)
        article.update(raw=raw, raw_sorted=sorted(raw, key=lambda x: (x.casefold(), x)),
                       raw_vocabulary=sorted(set(raw), key=lambda x: (x.casefold(), x)), normalized=normalized,
                       normalized_vocabulary=sorted(set(normalized)))
        articles.append(article)
    return articles


def fit_retrieval_models(documents: Sequence[str]) -> dict[str, tuple[Any, Any]]:
    """A3/A7/A8: Fit TF-IDF, count-unigram, and count-bigram models."""
    models = {
        "tfidf": TfidfVectorizer(min_df=2, max_df=.8, sublinear_tf=True),
        "count": CountVectorizer(min_df=2, max_df=.8),
        "bigram": CountVectorizer(ngram_range=(2, 2)),
    }
    return {name: (model, model.fit_transform(documents)) for name, model in models.items()}


def choose_query(raw_tokens: Sequence[str], vocabulary: set[str]) -> dict[str, Any]:
    """A4: Reproducibly sample three raw D1 tokens that survive normalization."""
    candidates = {}
    for raw in raw_tokens:
        normalized = preprocess_tokens([raw])
        if len(normalized) == 1 and normalized[0] in vocabulary:
            candidates.setdefault(normalized[0], raw)
    selected = random.Random(SEED).sample(list(candidates.items()), 3)
    return {"original": [raw for _, raw in selected],
            "normalized": [word for word, _ in selected], "operator": "AND", "seed": SEED}


def boolean_score(document: Sequence[str], query: Sequence[str]) -> float:
    """A5: Return 1 only when every AND-query term occurs."""
    return float(set(query).issubset(document))


def vector_score(model: Any, documents: Any, query_text: str) -> list[float]:
    """A6-A8: Transform the query and calculate cosine similarity."""
    query = model.transform([query_text])
    if query.nnz == 0:
        return [0.0] * documents.shape[0]
    return np.round(cosine_similarity(query, documents)[0], 6).tolist()


def expand_query(query: Sequence[str]) -> list[dict[str, Any]]:
    """A9: Add up to three same-POS WordNet synonyms per query term."""
    groups = []
    tag_to_pos = {"J": wn.ADJ, "V": wn.VERB, "N": wn.NOUN, "R": wn.ADV}
    for word, tag in pos_tag(list(query)):
        counts = Counter()
        for synset in wn.synsets(word, pos=tag_to_pos.get(tag[0])):
            for lemma in synset.lemmas():
                if lemma.name().isalpha() and lemma.name().lower() != word:
                    counts[lemma.name().lower()] += lemma.count()
        synonyms = [name for name, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:3]]
        groups.append({"original": word, "synonyms": synonyms, "alternatives": [word, *synonyms]})
    return groups


def expanded_boolean_score(document: Sequence[str], groups: Sequence[dict[str, Any]]) -> float:
    """A10: Use OR inside each synonym group and AND between groups."""
    vocabulary = set(document)
    return float(all(vocabulary.intersection(group["alternatives"]) for group in groups))


def expanded_query_text(groups: Sequence[dict[str, Any]], bigram: bool = False) -> str:
    if not bigram:
        return " ".join(term for group in groups for term in group["alternatives"])
    phrases = itertools.product(*(group["alternatives"] for group in groups))
    return " zzboundaryzz ".join(" ".join(phrase) for phrase in phrases)


def build_pos_vector(tokens: Sequence[str]) -> tuple[list[tuple[str, str]], list[int]]:
    """A11: Count tagged tokens on the fixed 45-tag Penn Treebank axes."""
    tagged = pos_tag(list(tokens))
    counts = Counter(tag for _, tag in tagged)
    return tagged, [counts.get(tag, 0) for tag in PENN_TAGS]


def pairwise_cosine_matrix(vectors: Sequence[Sequence[float]]) -> list[list[float]]:
    """A12: Calculate every pairwise cosine similarity."""
    return np.round(cosine_similarity(np.asarray(vectors, dtype=float)), 6).tolist()


def matrix_is_symmetric(matrix: Sequence[Sequence[float]]) -> bool:
    return bool(np.allclose(matrix, np.asarray(matrix).T))


# Section A compares the five news documents with several representations.
def run_section_a(articles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    documents = [" ".join(article["normalized"]) for article in articles]
    thresholds = []
    for low, high in ((1, 1.0), (1, .8), (2, 1.0), (2, .8)):
        trial = TfidfVectorizer(min_df=low, max_df=high).fit(documents)
        thresholds.append({"min_df": low, "max_df": high, "terms": len(trial.vocabulary_)})
    models = fit_retrieval_models(documents)
    tfidf, tfidf_docs = models["tfidf"]
    query = choose_query(articles[0]["raw"], set(tfidf.get_feature_names_out()))
    groups = expand_query(query["normalized"])
    base = {name: vector_score(model, matrix, " ".join(query["normalized"]))
            for name, (model, matrix) in models.items()}
    expanded = {name: vector_score(model, matrix, expanded_query_text(groups, name == "bigram"))
                for name, (model, matrix) in models.items()}
    expanded["boolean"] = [expanded_boolean_score(a["normalized"], groups) for a in articles]
    tagged_and_vectors = [build_pos_vector(article["normalized"]) for article in articles]
    return {
        "A1": [{"id": a["id"], "title": a["title"], "url": a["url"],
                "image_url": a["image_url"], "paragraph_count": a["paragraph_count"],
                "tokens": a["raw_sorted"], "vocabulary_size": len(a["raw_vocabulary"])}
               for a in articles],
        "A2": [{"id": a["id"], "token_count": len(a["normalized"]),
                "vocabulary_size": len(a["normalized_vocabulary"]),
                "vocabulary": a["normalized_vocabulary"]} for a in articles],
        "A3": {"thresholds": thresholds, "features": tfidf.get_feature_names_out().tolist(), "vectors": np.round(tfidf_docs.toarray(), 6).tolist()},
        "A4": query,
        "A5": [boolean_score(a["normalized"], query["normalized"]) for a in articles],
        "A6": base["tfidf"], "A7": base["count"], "A8": base["bigram"],
        "A9": groups, "A10": expanded,
        "A11": [{"tags": tagged, "vector": vector} for tagged, vector in tagged_and_vectors],
        "A12": pairwise_cosine_matrix([vector for _, vector in tagged_and_vectors]),
    }


def wordnet_neighborhood(word: str, synset: str | None = None) -> dict[str, Any]:
    """B1: Return the chosen noun sense, first hypernym, and all hyponyms."""
    selected = wn.synset(synset) if synset else wn.synsets(word, pos=wn.NOUN)[0]
    descendants = sorted(selected.closure(lambda synset: synset.hyponyms()), key=lambda x: x.name())
    hypernyms = selected.hypernyms()
    return {"synset": selected.name(), "hypernym": hypernyms[0].name() if hypernyms else None,
            "hyponyms": [synset.name() for synset in descendants]}


def wordnet_relations(word: str) -> dict[str, list[dict[str, Any]]]:
    """B2: Collect synonyms and direct antonyms, ranked by lemma count."""
    relations = {"synonyms": Counter(), "antonyms": Counter()}
    for synset in wn.synsets(word):
        for lemma in synset.lemmas():
            if lemma.name().lower() != word:
                relations["synonyms"][lemma.name().lower()] += lemma.count()
            for antonym in lemma.antonyms():
                relations["antonyms"][antonym.name().lower()] += antonym.count()
    return {kind: [{"word": name, "frequency": count} for name, count in
                   sorted(values.items(), key=lambda x: (-x[1], x[0]))]
            for kind, values in relations.items()}


def best_wordnet_similarity(first: str, second: str, metric: str, noun_only: bool = False) -> dict[str, Any]:
    """B3: Find the same-POS synset pair with the maximum WordNet score."""
    pos = wn.NOUN if noun_only else None
    candidates = []
    for left, right in itertools.product(wn.synsets(first, pos=pos), wn.synsets(second, pos=pos)):
        if left.pos() == right.pos() and (score := getattr(left, metric)(right)) is not None:
            candidates.append((float(score), left, right))
    if not candidates:
        return {"score": None, "first_synset": None, "second_synset": None,
                "least_common_subsumers": [], "first_hierarchy": [], "second_hierarchy": []}
    score, left, right = max(candidates, key=lambda row: (row[0], row[1].name(), row[2].name()))
    lcs = left.lowest_common_hypernyms(right)
    path = lambda synset: [item.name() for item in min(synset.hypernym_paths(), key=len)]
    return {"score": round(score, 6), "first_synset": left.name(), "second_synset": right.name(),
            "least_common_subsumers": [x.name() for x in lcs],
            "first_hierarchy": path(left), "second_hierarchy": path(right)}


def compare_relation_groups(first: str, second: str) -> list[dict[str, Any]]:
    """B4: Compare synonym/antonym combinations with path and Wu-Palmer."""
    left, right = wordnet_relations(first), wordnet_relations(second)
    groups = (("synonym-synonym", left["synonyms"], right["synonyms"]),
              ("antonym-antonym", left["antonyms"], right["antonyms"]),
              (f"{first} synonym-antonym", left["synonyms"], left["antonyms"]),
              (f"{second} synonym-antonym", right["synonyms"], right["antonyms"]))
    rows = []
    for group, first_words, second_words in groups:
        for a, b in itertools.product(first_words, second_words):
            path = best_wordnet_similarity(a["word"], b["word"], "path_similarity")
            wup = best_wordnet_similarity(a["word"], b["word"], "wup_similarity")
            rows.append({"group": group, "words": [a["word"], b["word"]],
                         "path": path["score"], "wup": wup["score"]})
    return rows


def direct_wordnet_similarity(first: str, second: str, metric: str) -> dict[str, Any]:
    """Compare two chosen synsets rather than searching all senses."""
    left, right = wn.synset(first), wn.synset(second)
    score = getattr(left, metric)(right)
    return {"score": round(float(score), 6), "first_synset": first, "second_synset": second,
            "least_common_subsumers": [x.name() for x in left.lowest_common_hypernyms(right)]}


def train_local_embeddings() -> dict[str, Any]:
    """B5/B6: Train reproducible Word2Vec and FastText models on Brown."""
    from gensim.models import FastText, Word2Vec
    sentences = [[w.lower() for w in sentence if WORD.fullmatch(w)] for sentence in brown.sents()]
    settings = dict(sentences=sentences, vector_size=100, window=5, min_count=2,
                    workers=1, sg=1, seed=SEED, epochs=10)
    return {"word2vec": Word2Vec(**settings).wv, "fasttext": FastText(**settings).wv}


def static_vector(term: str, model: Any) -> np.ndarray:
    vectors = [model.get_vector(word) for word in term.lower().split() if word in model]
    return np.mean(vectors, axis=0)


def bert_vector(term: str, tokenizer: Any, model: Any) -> np.ndarray:
    """B6: Mean-pool the non-special BERT subword states."""
    import torch
    encoded = tokenizer(term, return_tensors="pt")
    with torch.no_grad():
        states = model(**encoded).last_hidden_state[0]
    keep = [int(token) not in tokenizer.all_special_ids for token in encoded["input_ids"][0]]
    return states[keep].mean(0).numpy()


def embedding_result(words: Sequence[str], vector: Callable[[str], np.ndarray]) -> dict[str, Any]:
    """B5/B6/B8: Generate vectors and their cosine-similarity matrix."""
    vectors = [np.asarray(vector(word), dtype=float) for word in words]
    return {"vectors": {word: np.round(value, 8).tolist() for word, value in zip(words, vectors)},
            "similarity": pairwise_cosine_matrix(vectors)}


def wordnet_family() -> dict[str, Any]:
    """B7: Compare chosen dwelling senses with path and Wu-Palmer."""
    senses = [wn.synset(name) for name in FAMILY.values()]
    matrix = lambda metric: [[round(float(getattr(a, metric)(b) or 0), 6) for b in senses] for a in senses]
    return {"labels": list(FAMILY), "path": matrix("path_similarity"), "wup": matrix("wup_similarity")}


def compare_systems(wordnet: dict[str, Any], embeddings: dict[str, Any]) -> list[dict[str, Any]]:
    """B9: Use Spearman correlation to compare similarity rankings."""
    upper = lambda m: [m[i][j] for i in range(len(m)) for j in range(i + 1, len(m))]
    rows = []
    for metric in ("path", "wup"):
        for model, result in embeddings.items():
            rho = float(spearmanr(upper(wordnet[metric]), upper(result["similarity"])).statistic)
            rows.append({"wordnet": metric, "embedding": model,
                         "spearman_rho": None if math.isnan(rho) else round(rho, 6)})
    return rows


# Section B compares the chosen words with WordNet and four embeddings.
def run_section_b() -> dict[str, Any]:
    import gensim.downloader as api
    from transformers import AutoModel, AutoTokenizer
    local = train_local_embeddings()
    glove = api.load("glove-wiki-gigaword-100")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    bert = AutoModel.from_pretrained("google-bert/bert-base-uncased").eval()
    models = {**{name: (lambda word, model=model: static_vector(word, model)) for name, model in local.items()},
              "glove": lambda word: static_vector(word, glove),
              "bert": lambda word: bert_vector(word, tokenizer, bert)}
    pairs = {"required": REQUIRED_PAIR, "selected": PAIR}
    pair_senses = {"required": ("house.n.01", "train.n.01"),
                   "selected": ("potato.n.01", "tiger.n.02")}
    selected_senses = dict(zip(PAIR, pair_senses["selected"]))
    relations = {name: {word: wordnet_relations(word) for word in words} for name, words in pairs.items()}
    similarities = {name: {
        "maximum": {metric: best_wordnet_similarity(*words, metric + "_similarity", True)
                    for metric in ("path", "wup")},
        "selected": {metric: direct_wordnet_similarity(*pair_senses[name], metric + "_similarity")
                     for metric in ("path", "wup")},
    } for name, words in pairs.items()}
    pair_embeddings = {name: {model: embedding_result(words, vector) for model, vector in models.items()}
                       for name, words in pairs.items()}
    family_wordnet = wordnet_family()
    family_embeddings = {model: embedding_result(list(FAMILY), vector) for model, vector in models.items()}
    return {
        "pairs": {name: list(words) for name, words in pairs.items()},
        "B1": {name: {word: wordnet_neighborhood(word, selected_senses.get(word))
                       for word in words} for name, words in pairs.items()},
        "B2": relations, "B3": similarities,
        "B4": {name: compare_relation_groups(*words) for name, words in pairs.items()},
        "B5": {name: result["word2vec"] for name, result in pair_embeddings.items()},
        "B6": {name: {model: value for model, value in result.items() if model != "word2vec"}
               for name, result in pair_embeddings.items()},
        "B7": family_wordnet, "B8": family_embeddings,
        "B9": {"correlations": compare_systems(family_wordnet, family_embeddings),
               "conclusion": "WordNet measures taxonomic relations while embeddings measure contextual usage, so their rankings are complementary rather than identical."},
    }


def main() -> None:
    ensure_directories()
    ensure_nltk_data()
    results = {"section_a": run_section_a(load_articles()), "section_b": run_section_b()}
    output = ROOT / "outputs" / "submission_results.json"
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
