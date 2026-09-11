# Lab 2 defense notes

## Repository structure

- `lab2.py` is the submission file. It contains every NLP calculation for
  A1-A12 and B1-B9.
- `outputs/submission_results.json` is the complete output of `lab2.py`.
- `presentation.py` reads that output, extracts code examples, and prepares the
  website files. It performs no NLP calculations.
- `docs/` contains the GitHub Pages presentation.
- `tests/` checks important rules: three query terms, 45 POS dimensions,
  symmetric matrices, and the 400-line limit.

## How the script runs

1. Download the five articles and extract their body paragraphs.
2. Complete the Section A document-retrieval tasks.
3. Complete the Section B WordNet and embedding tasks.
4. Save every result in `outputs/submission_results.json`.
5. Run `presentation.py` separately to update the website.

## Section A - document retrieval

### A1 - extraction and raw tokens

The five BBC URLs are stored in the script. `requests` downloads each page with
a browser-style User-Agent, and BeautifulSoup selects the article paragraphs.
`word_tokenize()` produces the unfiltered token list. Sorting that list gives
the required alphabetical listing; counting distinct tokens gives the
document's vocabulary size.

### A2 - filtering and normalization

Only English-letter tokens longer than two characters are kept. They are
POS-tagged with their original capitalization, then lowercased; English
stopwords are removed before WordNet lemmatization. Original casing helps the
tagger distinguish proper nouns from other word types, and a word's correct
root form can depend on whether it is a noun, verb, adjective, or adverb.

Example: `cars` becomes `car`, while a verb such as `running` becomes `run`.

### A3 - TF-IDF vectors

Four `min_df` and `max_df` combinations are tested. The final model uses
`min_df=2`, so a word must occur in at least two documents, and `max_df=0.8`,
so a word cannot occur in all five documents. TF-IDF gives larger weights to
words that help distinguish one document from the others.

### A4 - three-word query

Three unfiltered tokens are taken from D1 after checking that their normalized
forms exist in the TF-IDF vocabulary. Seed `2026` makes the random selection
repeatable. Logical `AND` means all three concepts must occur for a Boolean
match.

### A5 - Boolean retrieval

The document and query are converted to sets. The expression
`set(query).issubset(document)` is true only when every query term occurs in the
document. A match is stored as `1.0`; a non-match is stored as `0.0`.

### A6 - TF-IDF retrieval

The fitted TF-IDF model transforms the query into the same vector space as the
documents. Cosine similarity then compares the query vector with each document
vector. Cosine uses vector direction, so document length has less influence.

### A7 - count-vector retrieval

`CountVectorizer` repeats the comparison using raw unigram counts instead of
TF-IDF weights. Comparing the rankings shows what changes when rare words are
no longer given extra importance.

### A8 - bigram retrieval

Another `CountVectorizer` uses `ngram_range=(2, 2)`. Each feature is now a pair
of adjacent words. This is stricter than unigram matching because both the
words and their order must match.

### A9 - WordNet query expansion

Each query word is POS-tagged. Compatible WordNet synonyms are ranked by lemma
frequency, and at most three are added. The original word remains in every
group, so expansion adds alternatives without replacing the initial query.

### A10 - retrieval with the expanded query

Boolean retrieval uses OR inside each synonym group and AND between the three
groups. The vector models transform the expanded query and recalculate cosine
similarity. `zzboundaryzz` separates alternative bigram phrases so no false
bigrams are created between alternatives. It must not be a stopword:
`CountVectorizer` would remove it before building bigrams and join the phrases
again.

### A11 - POS vectors

The normalized tokens are POS-tagged. Every document uses the same 45 Penn
Treebank tags as vector axes. Each component stores the number of tokens with
that tag.

### A12 - document similarity from grammar

Cosine similarity is calculated between every pair of 45-dimensional POS
vectors. A high value means similar grammatical distributions; it does not
necessarily mean that the documents discuss the same topic.

## Section B - word similarity

I selected `potato` and `tiger` for the word-level example. The required
`house` and `train` calculations are also included.

### B1 - synsets, hypernyms, and hyponyms

The first noun senses are retrieved for `house` and `train`. For the selected
pair, `potato.n.01` and `tiger.n.02` represent the intended plant-food and
animal meanings. The task then retrieves the first direct hypernym and all
descendant hyponyms.

### B2 - synonyms, antonyms, and frequency

Every WordNet synset and lemma is visited. Alternative lemma names become
synonyms; direct antonym links provide antonyms. Results are ranked with
`lemma.count()`, WordNet's tagged-corpus frequency. A zero means there are no
tagged occurrences for that lemma, not that the lemma is invalid.

### B3 - path and Wu-Palmer similarity

Words can have several meanings, so every same-POS synset pair is tested and
the pair with the highest score is kept. Path similarity uses distance through
the WordNet hierarchy. Wu-Palmer also considers the depth of the least common
subsumer (LCS). The explicit `potato.n.01` and `tiger.n.02` senses are calculated
separately to preserve the intended meanings.

### B4 - lexical-relation groups

Synonym-synonym, antonym-antonym, and synonym-antonym combinations are compared
with both WordNet measures. WordNet records no direct antonyms for many nouns,
including the selected words. An absent group is therefore left empty rather
than filled with invented zero-valued comparisons.

### B5 - Word2Vec

A 100-dimensional skip-gram Word2Vec model is trained on the NLTK Brown corpus.
One worker and seed `2026` make the training repeatable. Word2Vec learns vectors
from the contexts in which words occur. Cosine similarity compares two vectors.

### B6 - FastText, GloVe, and BERT

- FastText is trained on Brown with the same settings and also learns from
  character n-grams.
- GloVe supplies pretrained 100-dimensional vectors learned from global word
  co-occurrence statistics.
- BERT tokenizes each word, removes special tokens, and averages the remaining
  hidden states to produce one vector per term.

Cosine similarity is used for all three models so their results are comparable.

### B7 - WordNet dwelling-family table

The words are `house`, `apartment`, `flat`, `hotel`, `room`, `living room`, and
`toilet`. An explicit noun synset is chosen for each word to avoid unrelated
meanings. Seven words produce 7 by 7 path and Wu-Palmer matrices.

### B8 - embedding dwelling-family tables

The same 7 by 7 cosine-similarity matrix is generated for Word2Vec, FastText,
GloVe, and BERT. Static models represent a multiword term such as `living room`
by averaging its component-word vectors.

### B9 - compatibility conclusion

Only the upper triangle of each matrix is compared because the diagonal is
always one and the lower triangle repeats the same pairs. Spearman correlation
compares rankings instead of raw values. WordNet and embeddings are
complementary: WordNet represents taxonomic relations, while embeddings
represent contextual usage.

## Short answers to remember

- **Why does D1 match the Boolean query?** All three query terms were selected
  from D1, so their normalized forms occur there.
- **Why can a bigram score be zero?** The words may occur individually without
  appearing next to one another in the required order.
- **Why might expansion have little effect?** Added synonyms cannot affect a
  vector model when they are absent from the document vocabulary.
- **Why are POS similarities high?** News articles often use similar grammar
  even when their subjects differ.
- **Why did I choose `tiger.n.02`?** `tiger.n.01` means a fierce person;
  `tiger.n.02` is the animal intended for this comparison.
- **Why are many WordNet frequencies zero?** They are corpus occurrence counts,
  not similarity values and not proof that a lemma is missing.
- **Why do the models disagree?** They define similarity differently and learn
  from different information.
