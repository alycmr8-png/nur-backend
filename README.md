# Nur Backend API

FastAPI backend powered by your LangGraph Islamic AI agent (AnNur).

## Architecture

```
Nur App → Backend → LangGraph Agent
                        ├── jurisprudence_query (FAISS index of Islamic books)
                        ├── quran_search (Kalimat API)
                        ├── hadith_search (Kalimat API)
                        └── search_azkar (Kalimat API)
```

## Local Setup

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
```
Fill in `OPENAI_API_KEY` and set `FAISS_INDEX_PATH` to your local FAISS index folder.

### 4. Copy your FAISS index
Copy your `faiss_index_cleaned` folder into the `nur-backend` directory.

### 5. Run
```bash
uvicorn main:app --reload
```

Visit http://localhost:8000/docs for interactive API docs.

---

## Deploy to Railway

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Nur backend"
git remote add origin https://github.com/yourusername/nur-backend.git
git push -u origin main
```

### 2. Deploy on Railway
- Go to railway.app → New Project → Deploy from GitHub
- Select your repo

### 3. Add environment variables in Railway dashboard
- `OPENAI_API_KEY`
- `FAISS_INDEX_PATH` → set to `./faiss_index_cleaned`
- `KALIMAT_API_KEY`

### 4. Upload your FAISS index
The FAISS index must be included in your repo or uploaded separately.
Add the `faiss_index_cleaned` folder to your git repo (it's safe — no secrets in it).

### 5. Get your URL
Railway gives you a URL like `https://nur-backend-production.up.railway.app`
Update `BACKEND_URL` in `nur-app/src/constants/theme.js`

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/chat` | Ask Nur — full Islamic AI agent |
| POST | `/api/explain-verse` | Explain a Quran verse |
| POST | `/api/daily-lesson` | Generate daily Islamic lesson |
| POST | `/api/reflection-prompt` | Generate reflection prompt |

## Chat Request Format
```json
{
  "messages": [
    {"role": "user", "content": "What is the ruling on combining prayers?"}
  ],
  "thread_id": "user-123"
}
```
Use a unique `thread_id` per user to maintain conversation history.
