# Improved Approach — LLM Threat Modeling (MQ1–MQ4)

Full threat modeling pipeline extending the AISCC 2024 paper with six improvements.

---

## Six Improvements Over the Paper

| # | Improvement | Paper Limitation |
|---|-------------|-----------------|
| 1 | Full MQ1–MQ4 coverage | Paper only covers MQ1 & MQ2 |
| 2 | 3-layer Hallucination Guard | Paper acknowledges hallucinations but has no detection |
| 3 | Automated BLEU/ROUGE/BERTScore evaluation | Paper uses subjective binary +/- rating |
| 4 | Redis query cache (< 0.01 s on cache hit) | Paper has no caching (25–50 s per query) |
| 5 | Local LLM via Ollama (data stays on-premises) | Paper uses cloud models (data leaves machine) |
| 6 | Multi-format input: PDF, YAML, JSON, code, images | Paper accepts PDF only |

---

## Module Structure

```
improved_approach/
├── main.py                                      # CLI entry point — full MQ1–MQ4 pipeline
├── requirements.txt
├── README.md
│
├── document_loader/
│   └── multi_format_loader.py                  # IMP 6 — PDF/YAML/JSON/code/image loader
│
├── cache/
│   └── query_cache.py                          # IMP 4 — Redis cache with memory fallback
│
├── hallucination_guard/
│   └── hallucination_guard.py                  # IMP 2 — CVE + BERTScore + STRIDE validation
│
├── pipeline/
│   └── rag_pipeline_improved.py               # RAG with caching + Ollama LLM (IMP 4 + 5)
│
├── mq3_mitigation/
│   └── mq3_mitigation_generator.py            # IMP 1 — MQ3 NIST/OWASP mitigation plans
│
├── mq4_verifier/
│   └── mq4_verifier.py                        # IMP 1 — MQ4 weighted coverage checklist
│
└── evaluation/
    └── auto_evaluator.py                       # IMP 3 — BLEU/ROUGE-L/BERTScore evaluator
```

---

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama (for local LLM — recommended)
```bash
# Install Ollama: https://ollama.ai/download
ollama serve               # start the local LLM server
ollama pull llama3         # download Llama 3 (4.7 GB, one-time)
```

### 3. (Optional) Start Redis for caching
```bash
# Docker:
docker run -d -p 6379:6379 redis:alpine

# Or install locally: https://redis.io/docs/getting-started/
```

### 4. Configure environment variables
Create a `.env` file:
```
NVD_API_KEY=your_nvd_key        # free at https://nvd.nist.gov/developers/request-an-api-key
HUGGINGFACE_TOKEN=your_hf_token  # only needed if using --no-local (HuggingFace models)
```

---

## Usage

### Run full pipeline (MQ1 → MQ4) with local Llama 3
```bash
python main.py --input design.pdf --local
```

### Use Mistral instead of Llama 3
```bash
python main.py --input design.pdf --model mistral
```

### Multi-format input (PDF + Kubernetes YAML + architecture image)
```bash
python main.py --input design.pdf k8s-config.yaml architecture.png
```

### Custom MQ1 questions
```bash
python main.py --input design.pdf \
  --question "What APIs are externally exposed?" "How are secrets managed?"
```

### Use HuggingFace model instead of Ollama
```bash
python main.py --input design.pdf --no-local --model facebook/opt-125m
```

---

## Pipeline Overview

```
Input Files (PDF/YAML/JSON/code/image)
        │
        ▼
MultiFormatLoader.load()         [IMP 6 — multi-format]
        │
        ▼
RecursiveCharacterTextSplitter   [500-char chunks, 50-char overlap]
        │
        ▼
HuggingFaceEmbeddings → ChromaDB [all-MiniLM-L6-v2 → local vector store]
        │
   ┌────┴──────────────────┐
   ▼                        ▼
MQ1 Questions            MQ2 Questions
   │                        │
   └────────────┬───────────┘
                ▼
   ImprovedRAGPipeline.query()
        │
        ├──► QueryCache.get()          [IMP 4 — instant on cache hit]
        │         ▼ miss
        ├──► retriever.get_relevant_documents(k=4)
        ├──► LocalLLMClient.invoke()   [IMP 5 — Ollama, stays on-premises]
        ├──► HallucinationGuard:       [IMP 2]
        │       Layer 1: CVE regex → NVD API verification
        │       Layer 2: BERTScore vs RAG context
        │       Layer 3: STRIDE taxonomy check
        └──► QueryCache.set()
                │
                ▼
   MQ3MitigationGenerator.generate_all()   [IMP 1]
        │    builds NIST/OWASP prompt → LLM → parse JSON → verify CVEs
        ▼
   MQ4Verifier.verify_coverage()           [IMP 1]
        │    weighted checklist: STRIDE + assets + mitigations + CVEs
        ▼
   AutoEvaluator.evaluate_batch()          [IMP 3]
        │    BLEU + ROUGE-L + BERTScore vs ground truth
        ▼
   improved_results.json
```

---

## Performance vs Paper

| Metric | Paper | Improved |
|--------|-------|----------|
| MQ Coverage | MQ1 + MQ2 only | MQ1 + MQ2 + MQ3 + MQ4 |
| Response time (first query) | 25–50 s | 2–5 s (local Llama 3) |
| Response time (cached) | N/A | < 0.01 s |
| Hallucination detection | None | 3-layer guard |
| Evaluation method | Binary +/- (human) | BLEU / ROUGE-L / BERTScore |
| Input formats | PDF only | PDF, YAML, JSON, code, images |
| Data privacy | Cloud model | On-premises (Ollama) |
