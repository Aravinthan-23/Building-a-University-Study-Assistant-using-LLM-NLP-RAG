# University Study Assistant (Gemini + NLP + RAG + Streamlit)

## Features
- Upload multiple PDF lecture notes
- Extract and split PDF text into chunks
- Create semantic embeddings using `all-MiniLM-L6-v2`
- Retrieve relevant chunks using cosine similarity
- Generate grounded answers with Gemini
- Summaries, 2/7/15-mark exam answers, and MCQs
- English, Tamil-English (Thanglish), and Tamil answer options

## 1. Create a new project folder
Extract this ZIP, open the folder in VS Code, and open a terminal in this folder.

## 2. Create and activate a virtual environment (Windows)
```powershell
py -3.11 -m venv venv
.\venv\Scripts\activate
```

## 3. Install packages
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Add your Gemini API key
Create a file named `.streamlit/secrets.toml` and add:
```toml
GEMINI_API_KEY = "YOUR_NEW_GEMINI_API_KEY"
```
Use a newly generated key if your earlier key was exposed. Never upload `secrets.toml` to GitHub.

Alternatively, set an environment variable named `GEMINI_API_KEY`.

## 5. Run the app
```powershell
streamlit run app.py
```

## 6. Use the application
1. Upload one or more text-based PDF notes from the sidebar.
2. Click **Process PDF notes**.
3. Ask questions in the chat tab, or use Summary, Exam answers, and MCQ quiz.
4. The first run downloads the embedding model, so it may take a little time.

## Model configuration
The app currently uses `gemini-3.5-flash`, the model that was tested in the setup conversation. If your account later reports that the model is unavailable, change `MODEL_NAME` near the top of `app.py` to another generation model that appears in your `client.models.list()` output and works for your API key.

## Important limitations
- Scanned/image-only PDFs may not contain selectable text. OCR is not included in this starter version.
- The app uses in-memory session storage. Reprocess PDFs after restarting the app.
- The answer quality depends on the content and clarity of uploaded notes.
- Gemini API usage may incur charges or be subject to account quotas.
