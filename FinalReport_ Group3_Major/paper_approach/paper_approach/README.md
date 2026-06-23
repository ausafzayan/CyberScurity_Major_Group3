# Paper Approach — LLM Threat Modeling (MQ1 & MQ2)

Replication of the AISCC 2024 paper:
**"Facilitating Threat Modeling by Leveraging Large Language Models"**
(Elsharef, Zeng, Gu — University of Wisconsin-Milwaukee & IBM Research)

---

## Module Structure

```
paper_approach/
├── main.py                              # CLI entry point — orchestrates all modules
├── requirements.txt
├── document_processing/
│   └── document_processor.py           # PDF loading + text chunking (LangChain)
├── keyword_extraction/
│   └── keyword_extractor.py            # KeyBERT + NLTK keyword extraction (Task 2)
├── nvd_querier/
│   └── nvd_querier.py                  # NVD REST API → CVE dataset + JSONL save
├── vector_db/
│   └── vector_knowledge_base.py        # ChromaDB embed + store + retriever
├── rag_pipeline/
│   └── rag_pipeline.py                 # Llama 2 RAG chain (RetrievalQA)
└── evaluation/
    └── binary_evaluator.py             # Binary +/- evaluation (paper's method)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader stopwords punkt
```

### 2. Configure environment variables
Create a `.env` file in this directory:
```
NVD_API_KEY=your_nvd_api_key        # free at https://nvd.nist.gov/developers/request-an-api-key
HUGGINGFACE_TOKEN=your_hf_token     # required for Llama 2 gated model access
```

---

## Usage

### Task 1 — System Understanding (MQ1)
```bash
python main.py \
  --task 1 \
  --input design_document.pdf \
  --question "What components are involved in the design?"
```

### Task 2 — Threat Identification (MQ2)
```bash
python main.py \
  --task 2 \
  --input design_document.pdf \
  --question "How can shared private keys be exposed?"
```

### Use Llama 2 (paper's model — requires GPU + HuggingFace token)
```bash
python main.py \
  --task 1 \
  --input design.pdf \
  --question "What security measures does the system use?" \
  --model meta-llama/Llama-2-7b-chat-hf
```

### Multiple questions at once
```bash
python main.py --task 1 --input design.pdf \
  --question "What components are involved?" "How is data encrypted?" "What APIs are exposed?"
```

---

## Pipeline Overview

### Task 1 (MQ1) — System Understanding
```
PDF → DocumentProcessor.load_pdf()
    → split_documents() [500-char chunks, 20-char overlap]
    → VectorKnowledgeBase.build_from_documents() [HuggingFace embeddings → ChromaDB]
    → ThreatModelRAG.answer_question() [retrieve top-4 chunks → Llama 2 generation]
    → BinaryEvaluator.evaluate_response() [+/- rating]
```

### Task 2 (MQ2) — Threat Identification
```
PDF → full text
    → KeywordExtractor.extract_keywords() [KeyBERT top-15 keywords]
    → NVDQuerier.build_vulnerability_dataset() [NVD API → CVE records]
    → NVDQuerier.save_as_jsonl() [persist as JSONL]
    → VectorKnowledgeBase.build_from_cves() [embed CVE descriptions → ChromaDB]
    → ThreatModelRAG.answer_question() [retrieve top-5 CVE chunks → Llama 2]
    → BinaryEvaluator.evaluate_response() [+/- rating]
```

---

## Paper Results (reference)
| Metric | Value |
|--------|-------|
| Design documents evaluated | 12 |
| Total queries | 72 |
| Human satisfaction rate | 75%+ |
| Avg response time | 25–50 seconds |
| Base LLM avg words | 253 |
| RAG-enhanced avg words | 74 |

---

## Limitations (addressed in improved_approach/)
- Only covers MQ1 and MQ2 (no mitigation or coverage verification)
- Binary evaluation is subjective and not reproducible
- No hallucination detection
- Response time of 25–50 seconds
- PDF only — no YAML, JSON, code, or image input
- Cloud vector database required for scale (Pinecone)
