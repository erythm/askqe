# AskQE-BinaryQA: Binary Question Answering Extension for Machine Translation Quality Estimation

> **This repository is a fork of [AskQE](https://github.com/dayeonki/askqe) by Dayeon Ki, Kevin Duh, and Marine Carpuat (ACL 2025 Findings)**

I extend the original AskQE framework by introducing a **Binary Question Answering** approach that converts open-ended questions into Yes/No questions for more robust answer comparison.

---

## 🔄 What's New in This Extension

### Original AskQE vs Our Extension

| Aspect | Original AskQE | Our Extension (Binary QA) |
|--------|---------------|---------------------------|
| **Question Type** | Open-ended | Binary (Yes/No) |
| **Answer Comparison** | String similarity (F1, EM) | Exact match (Yes/No) |
| **Question Sources** | Atomic facts only | Template + SRL + Word Diff + Atomic |
| **Evaluation** | Answer overlap score | Weighted binary match score |

### Key Contributions

1. **Binary Question Generation**: Convert atomic facts into Yes/No questions for unambiguous answer comparison
2. **Multi-source Question Generation**:
   - **Template-based** (spaCy): Negation, numbers, dates, entities, verbs
   - **SRL-based** (LLM): Semantic roles (agent, patient, time, location)
   - **Word Difference**: Lexical divergence detection between source and back-translation
   - **Atomic Facts** (LLM): Fine-grained factual questions
3. **NLI Filtering**: Remove contradictory atomic facts using DeBERTa-based NLI
4. **Weighted Scoring**: Assign different weights to question types based on importance

---

## 📊 Results on BioMQM Dataset

| Metric | Original AskQE | Our Extension |
|--------|---------------|---------------|
| Kendall τ (vs MQM) | 0.171 | **0.162** |
| Spearman ρ (vs MQM) | ~0.19 | **0.206** |
| Decision Accuracy | ~50% | **59.2%** |
| Error Detection F1 | N/A | **0.626** |

### Per-Severity Detection Rate

| Severity | Detection Rate |
|----------|---------------|
| Critical | 75.0% |
| Major | **80.3%** |
| Minor | 52.9% |
| No Error | 53.9% |

---

## 🏗️ Pipeline Architecture
```
Source Text
     │
     ├──► [Template Questions] ──────────────────────┐
     │    (spaCy: negation, numbers, entities)       │
     │                                               │
     ├──► [SRL Questions] ───────────────────────────┤
     │    (LLM: agent, patient, time, location)      │
     │                                               ├──► Merge ──► Binary QA ──► Score
     ├──► [Word Difference Questions] ───────────────┤         (on Source & BT)
     │    (Lexical divergence detection)             │
     │                                               │
     └──► [Atomic Facts] ──► [NLI Filter] ──► [Atomic Questions]
          (LLM extraction)   (DeBERTa)      (LLM conversion)
```

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/erythm/askqe.git
cd askqe
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Requirements
```
vllm
transformers
accelerate
spacy
pandas
numpy
scipy
scikit-learn
matplotlib
```

### Running the Extension
```python
# See notebooks/Binary_extension.ipynb for full pipeline
```

---

## 📁 Repository Structure
```
askqe/
├── README.md
├── biomqm/
│   ├── dev_with_backtranslation.jsonl    # BioMQM dataset
│   └── askqe/
│       └── prompt.py                      # Prompt templates
├── QG/
│   └── code/
│       └── prompt.py                      # Question generation prompts
├── QA/
│   └── code/
│       └── prompt.py                      # Question answering prompts
├── evaluation/
│   └── string-comparison/
│       └── utils.py                       # Scoring utilities
├── extension/                             # 🆕 NEW
│   ├── Binary_extension.ipynb            # Main notebook
│   ├── atomic_facts_extracted.json       # Pre-extracted atomic facts
│   ├── extension_results.json            # Full results
│   └── extension_summary.csv             # Summary table
└── results/                               # 🆕 NEW
    └── results_visualization.png          # Result plots
```

---

## 📈 Methodology Details

### 1. Template-based Questions (No LLM)

Using spaCy, we extract:
- **Negations**: "Is the action 'given' negated?"
- **Numbers**: "Does the text mention '5mg'?"
- **Entities**: "Is 'Dr. Smith' mentioned?"
- **Dates**: "Does the text mention 'Monday'?"

### 2. SRL-based Questions (LLM)

Using Semantic Role Labeling:
- **Agent**: "Is 'the doctor' the one who performed the action?"
- **Patient**: "Is 'the patient' affected by the action?"
- **Negation**: "Is the action negated?"

### 3. Word Difference Questions (No LLM)

Detect lexical divergences:
```
Source: "All consecutive pats. with..."
BT:     "All consecutive patterns with..."

Question: "Does the text contain 'pats'?"
→ Source: Yes, BT: No → Mismatch detected!
```

### 4. Atomic Fact Questions (LLM + NLI)

1. Extract atomic facts from source
2. Filter contradictory facts using NLI
3. Convert to Yes/No questions

---

## 🔬 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Kendall τ** | Rank correlation with human severity ratings |
| **Spearman ρ** | Rank correlation (alternative) |
| **Decision Accuracy** | GMM-based accept/reject classification |
| **F1-Score** | Error detection performance |
| **ROC-AUC** | Classification quality |

---

## 📚 Citation

If you use this extension, please cite both the original paper and this work:
```bibtex
# Original AskQE Paper
@inproceedings{ki-etal-2025-askqe,
    title = "{A}sk{QE}: Question Answering as Automatic Evaluation for Machine Translation",
    author = "Ki, Dayeon and Duh, Kevin and Carpuat, Marine",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    year = "2025",
    url = "https://aclanthology.org/2025.findings-acl.899/",
}

# This Extension
@misc{askqe-binaryqa-2025,
    title = "AskQE-BinaryQA: Binary Question Answering Extension for MT Quality Estimation",
    author = "YOUR_NAME",
    year = "2025",
    url = "https://github.com/erythm/askqe",
}
```

---

## 🙏 Acknowledgments

- Original AskQE framework by [Dayeon Ki, Kevin Duh, Marine Carpuat](https://github.com/dayeonki/askqe)
- BioMQM dataset from WMT Biomedical Translation Task
- DeBERTa NLI model from Microsoft/Hugging Face

---

## 📧 Contact

For questions about this extension, please open an issue or contact [erfan.alerom@gmail.com].

---

## 📄 License

This project follows the same license as the original AskQE repository.
