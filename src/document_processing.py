# src/document_processing.py
import pdfplumber
import re
from langdetect import detect
import numpy as np
import fitz
from nltk.tokenize import sent_tokenize 
import nltk
nltk.download('punkt')
import time

nltk_data_path = r"D:\IMT\IMTM2S1\NLP\docurag\venv\Lib\site-packages\nltk_data"
nltk.data.path.append(nltk_data_path)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "Unknown"
    
def extract_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            text += f"\nPAGE {page_num+1}:\n{page.get_text()}\n"
    
    # Save raw extraction for inspection
    with open("debug_extracted.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    return text


def clean_text(self, text):
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        return re.sub(r'[^\w\s.,;:!?]', '', text) 
    
def _extract_pdf_text(pdf_path):
    """Replace existing PDF parsing implementation with this"""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
                
        # Debug: Print first 10000 characters
        print("=== RAW EXTRACTED TEXT ===")
        print(text[:10000])
        return text
    except Exception as e:
        print(f"PDF parsing failed: {str(e)}")
        return ""

def _extract_txt_text(file_path):
    """Simple text file reading"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text):
    """Text normalization"""
    # text = re.sub(r'\n+', ' ', text)
    # return re.sub(r'\s+', ' ', text).strip()
    cleaned = text.replace('\x00', '')  # Remove null bytes
    return cleaned.strip()[:100000]  # Limit text length

def chunk_text(text, max_chars=3000):
    
    """Chunk text into smaller pieces based on character count, preserving sentence boundaries."""
    if not text:
        return []
    
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def generate_chunk_summaries(text, summarizer, chunk_size=1500):
    """Generate summaries for text chunks then combine"""
    chunks = chunk_text(text)
    chunk_summaries = []
    
    for chunk in chunks:
        if len(chunk) > 100:  # Skip small chunks
            summary = summarizer.generate(chunk)
            chunk_summaries.append(summary)
    
    return " ".join(chunk_summaries)