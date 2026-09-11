# NLP Laboratory 2 — Semantic Similarity

A solution to the 2026 Semantic Similarity laboratory. The submission file,
`lab2.py`, contains the NLP calculations for every task and writes one JSON
result file.

## What is included

- Section A: five selected BBC articles, preprocessing, Boolean retrieval, TF–IDF,
  count vectors, bigrams, WordNet query expansion, and POS similarity.
- Section B: WordNet and embedding comparisons using the selected
  `potato`/`tiger` pair, plus the brief's required
  `house`/`train` and dwelling-family checks.
- Exact, line-linked implementation snippets inside every expanded web chapter.
- Deterministic query selection and static-embedding training with seed `2026`.

`lab2.py` stays below 400 lines and can be submitted and explained on its own.
It contains the article extraction, preprocessing, retrieval, WordNet,
embedding, and comparison logic. `presentation.py` only reads the JSON output
and prepares files for the website.

## Run it

Python 3.11 is recommended.

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python lab2.py
    python presentation.py
    python -m http.server 8000 -d docs

Open `http://localhost:8000` to view the site. The first command performs the
lab calculations; the second formats their output without repeating them.

## Reproducibility

The five source URLs and random seed are kept directly in `lab2.py`. The script
saves complete token lists, vocabularies, document vectors, POS tags, WordNet
results, embedding vectors, matrices, and comparison statistics in
`outputs/submission_results.json`.

Run tests with `python -m unittest discover -s tests -v`.

See [DEFENSE_GUIDE.md](DEFENSE_GUIDE.md) for the rationale behind each stage
and likely oral-defense questions. Generated artifacts are in `outputs/`;
the presentation is in `docs/`.

BBC article text and images remain the property of their respective owners.
The repository stores source links, derived statistics, and metadata; it does
not publish copied article bodies.
