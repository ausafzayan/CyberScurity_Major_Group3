# CyberSecurity_Major_Group3

> **Python version required: 3.11**

---

## 📋 Prerequisites

Before you begin, make sure the following are installed on your system:

| Tool | macOS | Windows |
|------|-------|---------|
| Python 3.11 | [python.org](https://www.python.org/downloads/) or `brew install python@3.11` | [python.org](https://www.python.org/downloads/) — ✅ check **"Add to PATH"** during install |
| Git | Pre-installed or `brew install git` | [git-scm.com](https://git-scm.com/download/win) |
| Ollama (local LLM) | [ollama.ai/download](https://ollama.ai/download) | [ollama.com/download/windows](https://ollama.com/download/windows) |

---

## ⚙️ Environment Setup

### Step 1 — Clone and navigate to the project folder

```bash
git clone git@github.com:ausafzayan/CyberScurity_Major_Group3.git
cd FinalReport_Group3_Major
```

### Step 2 — Verify Python 3.11

```bash
# macOS / Linux / Windows
python --version


> If the command is not found, ensure Python 3.11 is installed and added to your PATH.

### Step 3 — Create a virtual environment

```bash
# macOS / Linux / Windows
python -m venv .venv
```

### Step 4 — Activate the virtual environment

```bash
# macOS / Linux (bash/zsh)
source .venv/bin/activate

# Windows — Command Prompt
.venv\Scripts\activate.bat

# Windows — PowerShell
.venv\Scripts\Activate.ps1
```

> **Windows PowerShell note:** If you see a script execution error, run this once as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

You should now see `(.venv)` prefixed in your terminal prompt.

### Step 5 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6 — Set SSL certificates (macOS only)

```bash
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

> This step is only needed on macOS. Windows handles certificates automatically.

### Step 7 — Download NLTK data

```bash
python -m nltk.downloader stopwords punkt
```

---

## 🔑 Configure Environment Variables

Create a `.env` file inside `FinalReport_Group3_Major/`:

```bash
# macOS / Linux
touch .env

# Windows (Command Prompt)
type nul > .env

# Windows (PowerShell)
New-Item .env -ItemType File
```

Add the following to `.env`:

```env
NVD_API_KEY=your_nvd_api_key         # Free at https://nvd.nist.gov/developers/request-an-api-key
HUGGINGFACE_TOKEN=your_hf_token      # Required only for Llama 2 gated model access
```

---

## 🤖 Install and Start Ollama (Local LLM — Recommended)

Ollama runs a local LLM server so no API key or GPU is required for the default setup.

### macOS

```bash
# Option A: Download the app from https://ollama.ai/download
# Option B: Install via Homebrew
brew install ollama

# Terminal 1 — start the server
ollama serve

# Terminal 2 — download Llama 3 (one-time, ~4.7 GB)
ollama pull llama3
```

### Windows

1. Download and run `OllamaSetup.exe` from [ollama.com/download/windows](https://ollama.com/download/windows).
2. Click **Next → Install → Finish**. Ollama starts automatically in the system tray.
3. Open a new Command Prompt or PowerShell window and pull the model:

```bash
ollama pull llama3
```

> On Windows, `ollama serve` runs as a background service after installation — you do **not** need to run it manually.

---

## 🗄️ (Optional) Start Redis for Caching

Redis speeds up repeated queries by caching results locally.

```bash
# Using Docker (recommended — works the same on both platforms)
docker run -d -p 6379:6379 redis:alpine
```

If you don't have Docker:
- **macOS:** `brew install redis && brew services start redis`
- **Windows:** Download from [redis.io/docs/getting-started](https://redis.io/docs/getting-started/) or use the Windows port from [github.com/microsoftarchive/redis](https://github.com/microsoftarchive/redis/releases)

---

## 🚀 Usage

## paper_approach

```bash
cd paper_approach
```

**Task 1 — System Understanding (MQ1)**
```bash
python main.py \
  --task 1 \
  --input design_document.pdf \
  --question "What components are involved in the design?"
```

**Task 2 — Threat Identification (MQ2)**
```bash
python main.py \
  --task 2 \
  --input design_document.pdf \
  --question "How can shared private keys be exposed?"
```

**Use Llama 2 (paper's model — requires GPU + HuggingFace token)**
```bash
python main.py \
  --task 1 \
  --input design.pdf \
  --question "What security measures does the system use?" \
  --model meta-llama/Llama-2-7b-chat-hf
```

**Multiple questions at once**
```bash
python main.py --task 1 --input design.pdf \
  --question "What components are involved?" "How is data encrypted?" "What APIs are exposed?"
```

> **Windows note:** Replace `\` (line continuation) with `` ` `` in PowerShell, or write the full command on a single line in CMD.

---

## improved_approach

```bash
cd improved_approach
```

**Run full pipeline (MQ1 → MQ4) with local Llama 3**
```bash
python main.py --input design.pdf --local
```

**Use Mistral instead of Llama 3**
```bash
python main.py --input design.pdf --model mistral
```

**Multi-format input (PDF + Kubernetes YAML + architecture image)**
```bash
python main.py --input design.pdf k8s-config.yaml architecture.png
```

**Custom MQ1 questions**
```bash
python main.py --input design.pdf \
  --question "What APIs are externally exposed?" "How are secrets managed?"
```

**Use a HuggingFace model instead of Ollama**
```bash
python main.py --input design.pdf --no-local --model facebook/opt-125m
```

---

## 🛑 Deactivating the Virtual Environment

When you're done working, deactivate the venv:

```bash
deactivate
```

---

## ❓ Troubleshooting

| Problem | Fix |
|--------|-----|
| `python3.11: command not found` (macOS) | Run `brew install python@3.11` or use the full path `/usr/local/bin/python3.11` |
| `py -3.11` not found (Windows) | Reinstall Python 3.11 with **"Add to PATH"** and **"py launcher"** checked |
| SSL certificate errors (macOS) | Run `export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")` |
| PowerShell script blocked (Windows) | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` as Administrator |
| `ollama: command not found` | Restart the terminal after installing Ollama so PATH updates take effect |
| Redis connection refused | Start Redis with `docker run -d -p 6379:6379 redis:alpine` or skip Redis (optional) |
