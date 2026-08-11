# Web app hosting guide

## Architecture

- **Frontend**: React (Vite) in `frontend/`
- **Backend**: FastAPI in `backend/` — same search logic as the Streamlit app
- **Database**: Pinecone cloud (`topin-questions`)

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
  topin-app
```

Deploy the image to Render, Railway, Fly.io, or Azure Container Apps.

### Required env vars

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (default `topin-questions`)
- `PINECONE_CLOUD` / `PINECONE_REGION`
- `OPENROUTER_API_KEY` (optional, for friendlier intros)
- `HUGGINGFACEHUB_API_TOKEN` — required by the web backend. It calls the Hugging Face
  Inference API for query embeddings instead of loading the model locally (loading
  torch/sentence-transformers in-process needs 500MB+ RAM, which exceeds small hosting
  tiers like Render's 512MB plan). Get a free token at
  https://huggingface.co/settings/tokens.
- `data_link` — public URL to `topin_cleaned_data.csv` (used for tag index + topic catalog)

## Streamlit (legacy)

Still available:

```bash
streamlit run app.py
```
