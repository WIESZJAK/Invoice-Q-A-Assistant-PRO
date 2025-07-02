import streamlit as st
import sqlite3
import tempfile
import os
import easyocr
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from langdetect import detect
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import logging
import re
from datetime import datetime
from fuzzywuzzy import fuzz, process
from transformers import pipeline
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, filename='rag_pipeline.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
debug_logger = logging.getLogger('debug')
debug_handler = logging.FileHandler('debug_pipeline.log')
debug_handler.setLevel(logging.DEBUG)
debug_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
debug_logger.addHandler(debug_handler)

# Initialize QA model once
qa_model = pipeline("question-answering", model="deepset/bert-base-cased-squad2", device=0 if torch.cuda.is_available() else -1)
debug_logger.debug(f"QA model initialized on device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Initialize EasyOCR reader
try:
    reader = easyocr.Reader(['en', 'pl'])
except Exception as e:
    st.error(f"EasyOCR error: {e}. Install: `pip install easyocr torch torchvision opencv-python`")
    logger.error(f"EasyOCR init error: {e}")
    st.stop()

# Database setup
conn = sqlite3.connect("logs.db")
cursor = conn.cursor()
# Corrected CREATE TABLE statement
cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        confidence FLOAT,
        extracted_text TEXT,
        formatted_text TEXT,
        chunked_text TEXT,
        possible_outcomes TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS ocr_corrections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wrong TEXT NOT NULL,
        correct TEXT NOT NULL,
        UNIQUE(wrong, correct)
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS delimiters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        delimiter TEXT NOT NULL,
        UNIQUE(delimiter)
    )
""")
conn.commit()

# Enhanced OCR corrections from database
def correct_ocr_errors(text):
    cursor.execute("SELECT wrong, correct FROM ocr_corrections")
    corrections = dict(cursor.fetchall())
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    text = re.sub(r'(\d+)\s+(\d+)', r'\1\2', text)
    return text

# Image preprocessing with adjusted parameters
def preprocess_image(img):
    try:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save("debug_raw.jpg")
        logger.info("Raw image saved as debug_raw.jpg")
        
        img = img.convert('L')
        img_np = np.array(img)
        img_np = cv2.fastNlMeansDenoising(img_np, h=10)
        img_thresholded = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        img = Image.fromarray(img_thresholded)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img.save("debug_preprocessed.jpg")
        logger.info("Preprocessed image saved as debug_preprocessed.jpg")
        return img
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        return img.convert('L')

# Extract text with progress bar and live preview
def extract_text_from_image(path):
    try:
        img = Image.open(path)
        preprocessed_img = preprocess_image(img)
        st.image(preprocessed_img, caption="Live OCR Preview", width=500)
        st.write("Processing image...")
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        result = reader.readtext(np.array(preprocessed_img), detail=1, paragraph=True)
        progress_bar.progress(100)
        debug_logger.debug(f"OCR result: {result}")
        
        rows = {}
        tolerance = 35  # Increased for better tabular grouping
        for item in result:
            if len(item) == 2:
                bbox, text = item
                prob = 0.0
            elif len(item) == 3:
                bbox, text, prob = item
            else:
                debug_logger.warning(f"Skipping invalid OCR result: {item}")
                continue
            y_min, x_min = bbox[0][1], bbox[0][0]
            row_key = next((k for k in rows if abs(k - y_min) < tolerance), None)
            if row_key is None:
                row_key = y_min
                rows[row_key] = {}
            rows[row_key][x_min] = (text, prob)
        
        sorted_rows = sorted(rows.items(), key=lambda x: x[0])
        text_lines = [" ".join(item[1][0] for item in sorted(items[1].items(), key=lambda x: x[0])) for items in sorted_rows]
        text = "\n".join(text_lines)
        text = correct_ocr_errors(text)
        return text, text.split('\n')
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return "", []

# Format and clean text
def format_text(text):
    lines = text.split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            parts = re.split(r'\s+(?=\$?\d)', line)
            if parts:
                formatted_lines.append(" ".join(parts))
    return "\n".join(formatted_lines)

# Chunk text with dynamic delimiters
def get_vectorstore(text):
    try:
        if not text or len(text.strip()) < 10:
            logger.warning("Text too short for vector store")
            return None, text
        cursor.execute("SELECT delimiter FROM delimiters")
        delimiters = [row[0] for row in cursor.fetchall()]
        delimiter_pattern = r'(?i)(?:' + '|'.join(re.escape(d) for d in delimiters) + r')'
        sections = [s.strip() for s in re.split(delimiter_pattern, text) if s.strip()]
        docs = [Document(page_content=section) for section in sections]
        debug_logger.debug(f"Chunked into {len(docs)} sections: {[doc.page_content[:50] for doc in docs]}")
        if not docs:
            docs = [Document(page_content=text)]
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1")
        vector_store = FAISS.from_documents(docs, embeddings, distance_strategy="COSINE")
        chunked_text = "\n".join(f"{section} (Priority: {100 if any(d.lower() in section.lower() for d in delimiters) else 50}%)" for section in sections)
        return vector_store, chunked_text
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        return None, text

# Fuzzy matching for questions
def fuzzy_match(query, lines, threshold=80):
    corrected_query = query
    for word in query.split():
        best_match = process.extractOne(word, lines, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            corrected_query = corrected_query.replace(word, best_match[0])
    debug_logger.debug(f"Fuzzy match: '{query}' -> '{corrected_query}'")
    return corrected_query.lower()

# Extract answer with improved context handling
def extract_answer(text, question, lang):
    debug_logger.debug(f"Starting answer extraction for question: {question}")
    context = text
    corrected_question = fuzzy_match(question, text.split('\n'))
    debug_logger.debug(f"Corrected question: {corrected_question}")
    lines = text.split('\n')
    cursor.execute("SELECT delimiter FROM delimiters")
    delimiters = [row[0] for row in cursor.fetchall()]
    # Prioritize high-priority chunks based on delimiters, avoid index error
    chunk_lines = get_vectorstore(text)[1].split('\n')
    relevant_context = "\n".join(line for line in lines if any(d.lower() in line.lower() for d in delimiters) and any('100%' in cl for cl in chunk_lines))
    if not relevant_context and any(d.lower() in corrected_question.lower() for d in delimiters):
        # Extract first token from highest priority chunk if delimiter-related
        chunks = get_vectorstore(text)[1].split('\n')
        for chunk in chunks:
            if '100%' in chunk and any(d.lower() in chunk.lower() for d in delimiters):
                answer = chunk.split()[0]  # Take first word
                # Look for price pattern if delimiter suggests it, fix $ vs 8
                if any(d.lower() in ['price', 'total', 'amount'] for d in delimiters):
                    price_match = re.search(r'(?:8|\$)\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text)
                    if price_match:
                        amount = price_match.group()
                        if amount.startswith('8') and re.search(r'\btotal\b|\bamount\b', text, re.IGNORECASE):
                            answer = '$' + amount[1:] if amount[1:].replace(',', '').isdigit() else amount
                        else:
                            answer = amount
                return answer, 0.9  # High confidence for direct extraction
    relevant_context = relevant_context if relevant_context else context
    result = qa_model(question=corrected_question, context=relevant_context)
    answer = result['answer'] if result['score'] > 0.1 else "Not found"
    confidence = result['score'] if result['score'] > 0.1 else 0.0
    debug_logger.debug(f"QA result: answer={answer}, score={confidence}")
    return answer.strip(), min(confidence, 0.95)

# Streamlit app
st.set_page_config(page_title="Invoice Q&A Assistant PRO", layout="wide")
st.title("📄 Invoice Q&A Assistant (PRO Version)")

if 'text' not in st.session_state:
    st.session_state.text = None
if 'formatted_text' not in st.session_state:
    st.session_state.formatted_text = None
if 'chunked_text' not in st.session_state:
    st.session_state.chunked_text = None
if 'possible_outcomes' not in st.session_state:
    st.session_state.possible_outcomes = None
if 'interaction_id' not in st.session_state:
    st.session_state.interaction_id = None

with st.form(key='invoice_form'):
    uploaded_file = st.file_uploader("Upload invoice", type=["png", "jpg", "jpeg"])
    question = st.text_input("Ask a question:")
    submit_button = st.form_submit_button("Submit")

if uploaded_file and submit_button:
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    extracted_text, lines = extract_text_from_image(tmp_path)
    formatted_text = format_text(extracted_text)
    vector_store, chunked_text = get_vectorstore(formatted_text)
    st.session_state.text = extracted_text
    st.session_state.formatted_text = formatted_text
    st.session_state.chunked_text = chunked_text

    if not extracted_text.strip():
        st.warning("No text extracted.")
        logger.info(f"Q: {question}, A: Not found, C: 0.0")
        cursor.execute("INSERT INTO interactions (question, answer, confidence, extracted_text) VALUES (?, ?, ?, ?)",
                      (question, "Not found", 0.0, extracted_text))
        conn.commit()
    else:
        detected_lang = detect(extracted_text[:1000]) if extracted_text else 'en'
        if question:
            answer, confidence = extract_answer(formatted_text, question, detected_lang)
            possible_outcomes = f"{answer} (Confidence: {confidence:.2f})"
            st.session_state.possible_outcomes = possible_outcomes
            logger.info(f"Q: {question}, A: {answer}, C: {confidence}")
            cursor.execute("INSERT INTO interactions (question, answer, confidence, extracted_text, formatted_text, chunked_text, possible_outcomes) VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (question, answer, confidence, extracted_text, formatted_text, chunked_text, possible_outcomes))
            conn.commit()
            st.session_state.interaction_id = cursor.lastrowid

    os.unlink(tmp_path)

# Editable areas
if st.session_state.text:
    st.subheader("Editable Outputs")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        edited_extracted = st.text_area("Extracted Text", st.session_state.text, height=200, key="extracted")
    with col2:
        edited_formatted = st.text_area("Formatted Text", st.session_state.formatted_text, height=200, key="formatted")
    with col3:
        edited_chunked = st.text_area("Chunked Text", st.session_state.chunked_text, height=200, key="chunked")
    with col4:
        edited_outcomes = st.text_area("Possible Outcomes", st.session_state.possible_outcomes, height=200, key="outcomes")
    
    if st.button("Submit Corrections"):
        cursor.execute("""
            UPDATE interactions 
            SET extracted_text = ?, formatted_text = ?, chunked_text = ?, possible_outcomes = ?
            WHERE id = ?
        """, (edited_extracted, edited_formatted, edited_chunked, edited_outcomes, st.session_state.interaction_id))
        conn.commit()
        st.success("Corrections submitted successfully!")

# View Logs
if st.button("View Logs"):
    with open('rag_pipeline.log', 'r', encoding='utf-8', errors='replace') as log_file:
        logs = log_file.readlines()[-50:]
        st.text_area("Last 50 Log Entries", "\n".join(logs), height=300)

if st.session_state.possible_outcomes:
    st.subheader("ANSWER:")
    st.write(st.session_state.possible_outcomes)

# Interaction History
st.subheader("Interaction History")
with st.expander("See Interaction History", expanded=False):
    cursor.execute("SELECT question, answer, confidence, timestamp FROM interactions ORDER BY timestamp DESC")
    history = cursor.fetchall()
    if history:
        for row in history:
            st.write(f"**Q:** {row[0]} | **A:** {row[1]} | **Confidence:** {row[2]} | **Time:** {row[3]}")

conn.close()