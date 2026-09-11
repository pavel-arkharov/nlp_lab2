import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import lab2


class Lab2UnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lab2.ensure_directories()
        if str(lab2.NLTK_DATA) not in lab2.nltk.data.path:
            lab2.nltk.data.path.insert(0, str(lab2.NLTK_DATA))

    def test_bbc_article_extraction(self):
        body = "A complete BBC paragraph for repeatable parser testing. " * 12
        html = (
            '<html><head><meta property="og:title" content="Test title">'
            '<meta property="og:image" content="https://example.test/image.jpg">'
            '</head><body><article><div data-component="text-block"><p>'
            + body
            + '</p></div></article></body></html>'
        )
        article = lab2.extract_bbc_article(html, "D0", "https://example.test")
        self.assertEqual(article["title"], "Test title")
        self.assertEqual(article["paragraph_count"], 1)
        self.assertEqual(len(article["content_sha256"]), 64)

    def test_preprocessing_filters_and_lemmatizes(self):
        result = lab2.preprocess_tokens(
            ["The", "cars", "were", "running", ",", "in", "2026", "Paris"]
        )
        self.assertIn("car", result)
        self.assertIn("run", result)
        self.assertIn("paris", result)
        self.assertNotIn("the", result)
        self.assertNotIn(",", result)

    def test_preprocessing_pos_tags_before_lowercasing(self):
        tagged = [("The", "DT"), ("Paris", "NNP"), ("running", "VBG")]
        with patch("lab2.pos_tag", return_value=tagged) as tagger:
            result = lab2.preprocess_tokens(["The", "Paris", "running"])
        tagger.assert_called_once_with(["The", "Paris", "running"])
        self.assertEqual(result, ["paris", "run"])

    def test_boolean_requires_every_term(self):
        self.assertEqual(lab2.boolean_score(["one", "two", "three"], ["one", "three"]), 1.0)
        self.assertEqual(lab2.boolean_score(["one", "two"], ["one", "three"]), 0.0)

    def test_expanded_boolean_is_or_within_and_between(self):
        expansion = [
            {"alternatives": ["first", "foremost"]},
            {"alternatives": ["time", "sentence"]},
            {"alternatives": ["president", "chair"]},
        ]
        self.assertEqual(
            lab2.expanded_boolean_score(["foremost", "sentence", "chair"], expansion), 1.0
        )
        self.assertEqual(
            lab2.expanded_boolean_score(["foremost", "sentence"], expansion), 0.0
        )

    def test_bigram_expansion_uses_boundaries(self):
        expansion = [
            {"alternatives": ["a", "x"]},
            {"alternatives": ["b"]},
            {"alternatives": ["c", "y"]},
        ]
        text = lab2.expanded_query_text(expansion, bigram=True)
        self.assertEqual(text.count("zzboundaryzz"), 3)
        self.assertIn("a b c", text)
        self.assertIn("x b y", text)

    def test_bigram_boundary_prevents_cross_phrase_match(self):
        expansion = [
            {"alternatives": ["red", "green"]},
            {"alternatives": ["fox"]},
            {"alternatives": ["jumps", "sleeps"]},
        ]
        model = lab2.CountVectorizer(ngram_range=(2, 2)).fit(["jumps red"])
        safe_query = lab2.expanded_query_text(expansion, bigram=True)
        self.assertEqual(model.transform([safe_query]).nnz, 0)
        unsafe = lab2.CountVectorizer(ngram_range=(2, 2), stop_words=["zzboundaryzz"],
                                      vocabulary=model.vocabulary_)
        self.assertEqual(unsafe.transform([safe_query]).nnz, 1)

    def test_cosine_matrix_is_square_and_symmetric(self):
        matrix = lab2.pairwise_cosine_matrix([[1, 0], [0, 1], [1, 1]])
        self.assertEqual(np.asarray(matrix).shape, (3, 3))
        self.assertTrue(lab2.matrix_is_symmetric(matrix))
        self.assertEqual([row[i] for i, row in enumerate(matrix)], [1.0, 1.0, 1.0])

    def test_pos_vector_has_fixed_45_dimensions(self):
        _, vector = lab2.build_pos_vector(["house", "moves", "quickly"])
        self.assertEqual(len(lab2.PENN_TAGS), 45)
        self.assertEqual(len(vector), 45)
        self.assertEqual(sum(vector), 3)

    def test_wordnet_best_noun_pair_has_lcs(self):
        result = lab2.best_wordnet_similarity(
            "house", "train", "path_similarity", noun_only=True
        )
        self.assertGreater(result["score"], 0)
        self.assertTrue(result["least_common_subsumers"])
        self.assertTrue(result["first_synset"].endswith(".n.01"))


class GeneratedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path("docs/data/lab2_results.json")
        cls.results = json.loads(path.read_text(encoding="utf-8"))

    def test_generated_results_satisfy_invariants(self):
        a, b = self.results["section_a"], self.results["section_b"]
        self.assertEqual(len(a["A1"]), 5)
        self.assertEqual(len(a["A4"]["normalized"]), 3)
        self.assertTrue(all(len(item["vector"]) == 45 for item in a["A11"]))
        self.assertTrue(lab2.matrix_is_symmetric(a["A12"]))
        self.assertEqual(b["pairs"]["selected"], ["potato", "tiger"])

    def test_query_uses_three_distinct_raw_tokens(self):
        query = self.results["section_a"]["A4"]
        raw_tokens = self.results["section_a"]["A1"][0]["tokens"]
        self.assertEqual(len(set(query["normalized"])), 3)
        self.assertTrue(all(token in raw_tokens for token in query["original"]))

    def test_complete_raw_token_lists_are_exported_alphabetically(self):
        for article in self.results["section_a"]["A1"]:
            tokens = article["tokens"]
            self.assertEqual(tokens, sorted(tokens, key=lambda x: (x.casefold(), x)))

    def test_section_b_uses_selected_pair_and_brown_corpus(self):
        section = self.results["section_b"]
        self.assertEqual(section["pairs"]["selected"], ["potato", "tiger"])
        self.assertEqual(section["pairs"]["required"], ["house", "train"])
        self.assertEqual(list(section["B5"]["selected"]["vectors"]), ["potato", "tiger"])

    def test_every_assignment_task_has_live_source_snippet(self):
        expected = {f"A{i}" for i in range(1, 13)} | {f"B{i}" for i in range(1, 10)}
        snippets = self.results["code_snippets"]
        self.assertEqual(set(snippets), expected)
        for item in snippets.values():
            source = Path("lab2.py").read_text(encoding="utf-8")
            self.assertIn(item["function"], item["code"])
            self.assertIn("def " + item["function"], source)

    def test_submission_entry_point_stays_under_400_lines(self):
        line_count = len(Path("lab2.py").read_text(encoding="utf-8").splitlines())
        self.assertLessEqual(line_count, 400)

    def test_site_references_generated_data(self):
        script = Path("docs/app.js").read_text(encoding="utf-8")
        page = Path("docs/index.html").read_text(encoding="utf-8")
        self.assertIn('fetch("data/lab2_results.json?schema=20260914")', script)
        self.assertIn('id="section-a"', page)
        self.assertIn('id="section-b"', page)

    def test_article_cards_flip_between_source_and_corpus_statistics(self):
        script = Path("docs/app.js").read_text(encoding="utf-8")
        page = Path("docs/index.html").read_text(encoding="utf-8")
        styles = Path("docs/styles.css").read_text(encoding="utf-8")
        self.assertIn('shell.className="article-card-shell"', script)
        self.assertIn("normalized vocabulary", script.lower())
        self.assertIn("aria-pressed", script)
        self.assertIn(".article-card-shell.flipped .article-card-inner", styles)
        self.assertNotIn('id="article-detail"', page)

    def test_site_consolidates_section_a_and_labels_section_b(self):
        script = Path("docs/app.js").read_text(encoding="utf-8")
        page = Path("docs/index.html").read_text(encoding="utf-8")
        self.assertIn("SECTION_A_RESULTS", script)
        self.assertIn("final table collects", script)
        self.assertIn("potato ↔ tiger", script)
        self.assertIn("Section A · Document-retrieval workflow", page)
        self.assertNotIn("From text to evidence", page)
        self.assertIn("Their WordNet relations", page)


if __name__ == "__main__":
    unittest.main()
