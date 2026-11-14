import os
import re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(page_title="University Study Assistant", page_icon="🎓", layout="wide")

MODEL_NAME = "gemini-3.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def get_api_key():
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

def extract_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {page_number}]\n{text}")
    return "\n\n".join(pages)

def split_into_chunks(text, chunk_size=900, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks

def retrieve_chunks(question, chunks, embeddings, model, top_k=4):
    query_embedding = model.encode([question], normalize_embeddings=True)[0]
    scores = np.dot(embeddings, query_embedding)
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best_indices]

def ask_gemini(client, prompt):
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text or "The model returned an empty response."

def make_context(results):
    return "\n\n".join(
        f"Source {i+1} (similarity: {score:.3f}):\n{chunk}"
        for i, (chunk, score) in enumerate(results)
    )

st.title("🎓 University Study Assistant")
st.caption("Ask questions from your notes using Gemini + NLP embeddings + Retrieval-Augmented Generation (RAG).")

api_key = get_api_key()
if not api_key:
    st.warning("Add your Gemini API key before using the assistant.")
    st.code('GEMINI_API_KEY = "YOUR_NEW_API_KEY"', language="toml")
    st.info("Save this in .streamlit/secrets.toml, or set the GEMINI_API_KEY environment variable. Do not paste your key into app.py or share it.")
    st.stop()

client = genai.Client(api_key=api_key)
embedding_model = load_embedding_model()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "document_names" not in st.session_state:
    st.session_state.document_names = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("📚 Study materials")
    uploaded_files = st.file_uploader(
        "Upload your lecture notes (PDF)",
        type=["pdf"],
        accept_multiple_files=True
    )
    if st.button("Process PDF notes", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            all_chunks = []
            names = []
            with st.spinner("Reading PDFs and creating embeddings..."):
                for pdf in uploaded_files:
                    pdf_text = extract_pdf_text(pdf)
                    if pdf_text.strip():
                        pdf_chunks = split_into_chunks(pdf_text)
                        all_chunks.extend([f"Document: {pdf.name}\n{c}" for c in pdf_chunks])
                        names.append(pdf.name)
                if all_chunks:
                    vectors = embedding_model.encode(
                        all_chunks,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    st.session_state.chunks = all_chunks
                    st.session_state.embeddings = np.asarray(vectors, dtype=np.float32)
                    st.session_state.document_names = names
                    st.session_state.chat_history = []
                    st.success(f"Processed {len(names)} PDF(s) into {len(all_chunks)} text chunks.")
                else:
                    st.error("No selectable text was found. This may be a scanned PDF; run OCR on it first.")

    if st.session_state.document_names:
        st.success("Knowledge base ready")
        for name in st.session_state.document_names:
            st.caption("• " + name)
        if st.button("Clear uploaded knowledge base", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.embeddings = None
            st.session_state.document_names = []
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.caption(f"LLM: {MODEL_NAME}")
    st.caption("Embeddings: all-MiniLM-L6-v2")

tab_chat, tab_summary, tab_exam, tab_quiz = st.tabs(
    ["💬 Ask your notes", "📝 Summarize", "📖 Exam answers", "🧠 MCQ quiz"]
)

with tab_chat:
    st.subheader("Chat with your study materials")
    if not st.session_state.chunks:
        st.info("Upload one or more PDFs in the sidebar and click **Process PDF notes**.")
    else:
        for item in st.session_state.chat_history:
            with st.chat_message(item["role"]):
                st.markdown(item["content"])

        question = st.chat_input("Ask a question from your uploaded notes...")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Searching your notes and generating an answer..."):
                    retrieved = retrieve_chunks(
                        question,
                        st.session_state.chunks,
                        st.session_state.embeddings,
                        embedding_model
                    )
                    context = make_context(retrieved)
                    prompt = f"""
You are a helpful University Study Assistant.
Answer the student's question using the supplied study-note context.
If the answer is not present in the context, clearly say that the notes do not contain enough information.
Do not invent facts. Explain in simple, student-friendly English. Use headings or bullet points where useful.

STUDY NOTE CONTEXT:
{context}

STUDENT QUESTION:
{question}
"""
                    try:
                        answer = ask_gemini(client, prompt)
                    except Exception as e:
                        answer = f"Gemini request failed: {e}"
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    with st.expander("Retrieved note sections"):
                        for i, (chunk, score) in enumerate(retrieved, start=1):
                            st.markdown(f"**Section {i} · similarity {score:.3f}**")
                            st.write(chunk)

with tab_summary:
    st.subheader("Summarize your notes")
    summary_style = st.selectbox("Summary type", ["Quick revision notes", "Detailed summary", "Important points", "Explain in simple English"])
    if st.button("Generate summary", key="summary_btn", disabled=not bool(st.session_state.chunks)):
        with st.spinner("Creating summary from your notes..."):
            context = "\n\n".join(st.session_state.chunks[:12])
            prompt = f"Create {summary_style.lower()} from these university study notes. Keep it accurate and organized with headings and bullet points. Do not add unsupported facts.\n\nNOTES:\n{context}"
            try:
                st.markdown(ask_gemini(client, prompt))
            except Exception as e:
                st.error(f"Gemini request failed: {e}")
    if not st.session_state.chunks:
        st.info("Process PDF notes first.")

with tab_exam:
    st.subheader("Prepare exam answers")
    topic = st.text_input("Enter the topic", placeholder="Example: Data Drift and Concept Drift")
    marks = st.selectbox("Answer format", ["2 marks", "7 marks", "15 marks"])
    language = st.selectbox("Explanation language", ["Simple English", "Tamil-English (Thanglish)", "Tamil"])
    if st.button("Generate exam answer", key="exam_btn"):
        if not topic.strip():
            st.warning("Enter a topic.")
        elif not st.session_state.chunks:
            st.warning("Upload and process your notes first.")
        else:
            with st.spinner("Preparing your exam answer..."):
                retrieved = retrieve_chunks(topic, st.session_state.chunks, st.session_state.embeddings, embedding_model)
                context = make_context(retrieved)
                prompt = f"""
Prepare a university exam answer for: {topic}
Required length/format: {marks}
Language: {language}
Use this retrieved study material as the main source. Explain clearly, use suitable headings and examples, and do not invent details missing from the notes.
For 2 marks: concise definition and key point.
For 7 marks: introduction, explanation, example, conclusion.
For 15 marks: detailed structured answer with suitable subheadings, examples, advantages/limitations where relevant, and conclusion.

STUDY MATERIAL:
{context}
"""
                try:
                    st.markdown(ask_gemini(client, prompt))
                except Exception as e:
                    st.error(f"Gemini request failed: {e}")

with tab_quiz:
    st.subheader("Generate MCQs from your notes")
    number_of_questions = st.slider("Number of questions", 3, 15, 5)
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    if st.button("Generate MCQs", key="quiz_btn", disabled=not bool(st.session_state.chunks)):
        with st.spinner("Generating practice questions..."):
            context = "\n\n".join(st.session_state.chunks[:15])
            prompt = f"""
Create {number_of_questions} {difficulty.lower()} multiple-choice questions based only on these study notes.
For each question, provide four options (A-D), the correct answer, and a one-sentence explanation.
Avoid duplicate questions. Format clearly.

NOTES:
{context}
"""
            try:
                st.markdown(ask_gemini(client, prompt))
            except Exception as e:
                st.error(f"Gemini request failed: {e}")
    if not st.session_state.chunks:
        st.info("Process PDF notes first.")

st.divider()
st.caption("RAG pipeline: PDF text extraction → text chunking → sentence embeddings → cosine-similarity retrieval → Gemini answer generation.")
