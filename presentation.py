#!/usr/bin/env python3
"""Turn lab2.py output into files read by the presentation website."""

import inspect
import json
import textwrap
from pathlib import Path

import lab2

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "outputs" / "submission_results.json"
SITE_DATA = ROOT / "docs" / "data" / "lab2_results.json"

TASK_FUNCTIONS = {
    "A1": lab2.extract_bbc_article,
    "A2": lab2.preprocess_tokens,
    "A3": lab2.fit_retrieval_models,
    "A4": lab2.choose_query,
    "A5": lab2.boolean_score,
    "A6": lab2.vector_score,
    "A7": lab2.fit_retrieval_models,
    "A8": lab2.fit_retrieval_models,
    "A9": lab2.expand_query,
    "A10": lab2.expanded_boolean_score,
    "A11": lab2.build_pos_vector,
    "A12": lab2.pairwise_cosine_matrix,
    "B1": lab2.wordnet_neighborhood,
    "B2": lab2.wordnet_relations,
    "B3": lab2.best_wordnet_similarity,
    "B4": lab2.compare_relation_groups,
    "B5": lab2.train_local_embeddings,
    "B6": lab2.bert_vector,
    "B7": lab2.wordnet_family,
    "B8": lab2.embedding_result,
    "B9": lab2.compare_systems,
}


def code_snippets() -> dict:
    snippets = {}
    for task, function in TASK_FUNCTIONS.items():
        lines, start = inspect.getsourcelines(function)
        shown = lines[:42]
        if len(lines) > 42:
            shown.append("    # ... remainder omitted from the presentation\n")
        end = start + min(len(lines), 42) - 1
        snippets[task] = {
            "function": function.__name__,
            "line_start": start,
            "line_end": end,
            "code": textwrap.dedent("".join(shown)).rstrip(),
            "source_url": f"https://github.com/pavel-arkharov/nlp_lab2/blob/main/lab2.py#L{start}-L{end}",
        }
    return snippets


def markdown_report(results: dict) -> str:
    a, b = results["section_a"], results["section_b"]
    lines = ["# Laboratory 2 - Semantic Similarity", "", "## Section A"]
    for raw, normalized in zip(a["A1"], a["A2"]):
        lines.append(
            f"- **{raw['id']}**: {len(raw['tokens'])} raw tokens, "
            f"{raw['vocabulary_size']} raw vocabulary items, "
            f"{normalized['vocabulary_size']} normalized vocabulary items"
        )
    lines += ["", f"Query: **{' AND '.join(a['A4']['original'])}**", "",
              f"Normalized query: `{' AND '.join(a['A4']['normalized'])}`", "",
              "## Section B", "",
              "Selected words: **potato** and **tiger**.", "",
              b["B9"]["conclusion"], ""]
    return "\n".join(lines)


def main() -> None:
    results = json.loads(INPUT.read_text())
    results["code_snippets"] = code_snippets()
    serialized = json.dumps(results, indent=2, ensure_ascii=False) + "\n"
    SITE_DATA.parent.mkdir(parents=True, exist_ok=True)
    SITE_DATA.write_text(serialized)
    (ROOT / "outputs" / "lab2_results.json").write_text(serialized)
    report = markdown_report(results)
    (ROOT / "outputs" / "lab2_report.md").write_text(report)
    download = ROOT / "docs" / "downloads" / "lab2_report.md"
    download.parent.mkdir(parents=True, exist_ok=True)
    download.write_text(report)
    print(f"Website data: {SITE_DATA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
