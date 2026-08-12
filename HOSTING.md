# Web app hosting guide

## Architecture

- **Frontend**: React (Vite) in `frontend/`
- **Backend**: FastAPI in `backend/` — same search logic as the Streamlit app
- **Database**: Pinecone cloud (`topin-questions`)

## Important: deploy branch

The FastAPI + React app lives on branch **`new`**.

Branch **`main`** is still the old Streamlit-only app.  
If Render is set to `main`, your pushes to `new` will **not** go live.

In Render → Settings → Build & Deploy → **Branch = `new`**.

## Local development

### 1. Install Python deps

```bash
pip install -r requirements.txt -r requirements-web.txt
```

### 2. Install & run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173 (proxies `/api` to backend)

### 3. Run backend

```bash
# from project root
set TOPIN_API_MODE=1
set EMBEDDINGS_BACKEND=fast
uvicorn backend.main:app --reload --port 8000
```

## Production / hosting

### Build frontend into backend

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

FastAPI serves `frontend/dist` automatically when present.

### Docker

```bash
docker build -t topin-app .
docker run -p 8000:8000 ^
  -e PINECONE_API_KEY=your_key ^
  -e PINECONE_INDEX_NAME=topin-questions ^
  -e PINECONE_CLOUD=aws ^
  -e PINECONE_REGION=us-east-1 ^
  -e OPENROUTER_API_KEY=optional ^
  -e data_link=https://your-csv-url ^
  topin-app
```

### Render (recommended)

1. Push branch **`new`** to GitHub.
2. New Web Service → connect `Topin-RAG` → **Docker**.
3. Set **Branch** to **`new`** (not `main`).
4. Health check path: `/api/health`.
5. Add env vars:

| Key | Value |
|-----|--------|
| `PINECONE_API_KEY` | your Pinecone key |
| `PINECONE_INDEX_NAME` | `topin-questions` |
| `PINECONE_CLOUD` | `aws` |
| `PINECONE_REGION` | `us-east-1` |
| `EMBEDDINGS_BACKEND` | `fast` |
| `SKIP_LLM_INTRO` | `1` |
| `data_link` | public URL to `topin_cleaned_data.csv` |
| `OPENROUTER_API_KEY` | optional |

6. Use at least **Starter** (512MB free often OOMs during Docker build/runtime).
7. After deploy, open `https://YOUR-SERVICE.onrender.com/api/health` — should return `{"ok": true, ...}`.

Do **not** set `EMBEDDINGS_BACKEND=local` on Render free/starter — torch MiniLM needs ~1GB+ RAM and will crash the service.

### Required env vars

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default `topin-questions`)
- `PINECONE_CLOUD` / `PINECONE_REGION`
- `OPENROUTER_API_KEY` (optional, for friendlier intros)
- `EMBEDDINGS_BACKEND` — `fast` (default, Render-safe), `local`, or `remote`
  - **fast**: ONNX MiniLM via `fastembed` (low RAM, no HF Inference token)
  - **local**: `sentence-transformers` + torch (~1GB+ RAM)
  - **remote**: Hugging Face Inference API — token must allow Inference access
- `SKIP_LLM_INTRO=1` — skips OpenRouter intro text for faster responses
- `data_link` — public URL to `topin_cleaned_data.csv` (tag index + topic catalog)

## Streamlit (legacy)

Still available:

```bash
streamlit run app.py
```
