# UzLegalQA: Benchmark Dataset for Uzbek Legal Information Retrieval

[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Sohiba01%2Fuzbekh--legal--ir-blue)](https://huggingface.co/datasets/Sohiba01/uzbek-legal-ir)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)

> **The first open benchmark dataset for statutory law retrieval in the Uzbek language.**

UzLegalQA contains **4,943 article-level semantic chunks** from 8 major statutory codes of the Republic of Uzbekistan and **200 intent-annotated user queries** across 4 categories. It provides the foundational infrastructure for developing and evaluating legal information retrieval (IR) and retrieval-augmented generation (RAG) systems in Uzbek.

---

## 📋 Dataset Overview

| Property | Value |
|----------|-------|
| Language | Uzbek (uz) |
| Legal system | Civil law (statutory codes) |
| Total chunks | 4,943 |
| Total queries | 200 |
| Annotated queries | 50 (graded relevance) |
| Legal codes | 8 |
| Query intent categories | 4 |
| HuggingFace | [Sohiba01/uzbek-legal-ir](https://huggingface.co/datasets/Sohiba01/uzbek-legal-ir) |

---

## 📚 Corpus — Statutory Codes

| Code | Domain | Chunks |
|------|--------|--------|
| Civil Code | Civil law | 485 |
| Criminal Code | Criminal law | 601 |
| Administrative Code | Administrative liability | 1,426 |
| Labor Code | Labor law | 818 |
| Family Code | Family law | 262 |
| Electoral Code | Electoral law | 126 |
| Tax Code | Tax law | 1,056 |
| Housing Code | Housing law | 169 |
| **Total** | | **4,943** |

---

## 🔍 Query Intent Categories

| Intent | Count | Example |
|--------|-------|---------|
| Procedural | 60 (30%) | *"Ishdan bo'shatish tartibi qanday?"* |
| Factual | 66 (33%) | *"Mehnat shartnomasi nima?"* |
| Consequence | 45 (22.5%) | *"Shartnomani buzsa nima bo'ladi?"* |
| Temporal | 29 (14.5%) | *"Ta'til muddati qancha?"* |

---

## 📊 Baseline Results (BM25, 50 annotated queries)

| Intent | nDCG@10 | MRR | MAP | R@10 | P@5 |
|--------|---------|-----|-----|------|-----|
| Procedural | 0.7332 | 0.7222 | 0.6420 | 0.9333 | 0.4667 |
| Factual | 0.8513 | 0.8222 | 0.8253 | 1.0000 | 0.7467 |
| Consequence | 0.8207 | 0.9167 | 0.8509 | 0.9167 | 0.6500 |
| Temporal | 0.8749 | 1.0000 | 0.8894 | 1.0000 | 0.7000 |
| **Overall** | **0.8123** | **0.8433** | **0.7867** | **0.9600** | **0.6320** |

Relevance scale: `0` = non-relevant · `1` = partially relevant · `2` = fully relevant

---

## 📁 Repository Structure

```
uzLegalQA_benchmark-dataset/
├── data/
│   ├── UzLegalQA_Scored.xlsx      # 50-query relevance judgments (auto-scored)
│   └── UzLegalQA_Annotation.xlsx  # Full annotation template
├── evaluation/
│   ├── calculate_metrics.py       # nDCG@10, MRR, MAP, R@10, P@5
│   └── UzLegalQA_Metrics.xlsx     # Pre-computed metric results
├── baselines/
│   └── create_annotation.py       # BM25 annotation generation script
└── README.md
```

---

## 🚀 Quick Start

### Load dataset from HuggingFace

```python
from datasets import load_dataset

dataset = load_dataset("Sohiba01/uzbek-legal-ir")
print(dataset["train"][0])
# {'text': 'Mehnat kodeksi. 1-modda. ...'}
```

### Run evaluation metrics

```bash
pip install pandas openpyxl numpy

python calculate_metrics.py --input UzLegalQA_Scored.xlsx --output results.json
```

**Output:**
```
---------------------------------------------------------------
Intent           nDCG@10        MRR        MAP       R@10        P@5
---------------------------------------------------------------
Procedural        0.7332     0.7222     0.6420     0.9333     0.4667
Factual           0.8513     0.8222     0.8253     1.0000     0.7467
Consequence       0.8207     0.9167     0.8509     0.9167     0.6500
Temporal          0.8749     1.0000     0.8894     1.0000     0.7000
---------------------------------------------------------------
OVERALL           0.8123     0.8433     0.7867     0.9600     0.6320
---------------------------------------------------------------
```

---

## 🔑 Key Findings

- **BM25 achieves strong overall performance** (nDCG@10 = 0.8123) for the majority of queries
- **Temporal queries** reach perfect MRR (1.0000) — time-specific legal terms map directly onto statutory text
- **Procedural queries** show the weakest performance (nDCG@10 = 0.7332) due to vocabulary mismatch
- **Two complete retrieval failures** identified (Q006: *meros* vs. *vorislik*; Q035: *trafik* vs. *yo'l harakati xavfsizligi*) — motivating LLM-based query expansion as future work

---

## 📖 Citation

If you use UzLegalQA in your research, please cite:

```bibtex
@article{yusupova2025uzlegalqa,
  title     = {UzLegalQA: A Benchmark Dataset and Baseline Evaluation 
               for Uzbek Legal Information Retrieval},
  author    = {Yusupova, Sohiba},
  journal   = {Information Processing \& Management},
  year      = {2025},
  note      = {Under review}
}
```

---

## 🔗 Links

- 🤗 **HuggingFace Dataset:** [Sohiba01/uzbek-legal-ir](https://huggingface.co/datasets/Sohiba01/uzbek-legal-ir)
- 📜 **Source corpus:** [lex.uz](https://lex.uz) — Official legal portal of the Republic of Uzbekistan

---

## 📄 License

This dataset and code are released under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

Statutory texts sourced from [lex.uz](https://lex.uz), the official legal information portal of the Republic of Uzbekistan. Baseline retrieval implemented using [rank-bm25](https://github.com/dorianbrown/rank_bm25) and [sentence-transformers](https://www.sbert.net/).
