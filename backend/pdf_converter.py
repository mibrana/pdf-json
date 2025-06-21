import json
import re
import logging
from typing import List, Dict, Union, Optional
from pathlib import Path

import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def simple_paragraph_tokenize(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return paragraphs

def chunk_text_paragraphs(paragraphs: List[str], max_chars: int = 1000) -> List[str]:
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(para) < 20:
            continue
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += " " + para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

class PDFToJSON:
    def __init__(self, ocr_enabled: bool = True, ollama_model: str = "gemma2:2b"):
        self.ocr_enabled = ocr_enabled
        self.ollama_model = ollama_model
        self.structure_patterns = {
            'heading': r'^[A-Z][A-Z\s\.\-\']{3,}$',
            'date': r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'phone': r'\+?[\d\-\(\)\s]{10,}',
            'amount': r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?',
            'list_item': r'^\s*[\u2022\-\*]\s',
        }
        logger.info(f"Initialized PDFToJSON with OCR={'ON' if ocr_enabled else 'OFF'} and Ollama model '{self.ollama_model}'.")

    def extract_text_with_formatting(self, pdf_path: Union[str, Path]) -> List[Dict]:
        elements = []
        if self.ocr_enabled:
            logger.info("Using OCR to extract text from scanned PDF...")
            try:
                images = convert_from_path(pdf_path)
                for page_number, image in enumerate(images, start=1):
                    logger.info(f"Performing OCR on page {page_number}")
                    image_cv = np.array(image)
                    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    preprocessed_image = Image.fromarray(thresh)
                    text = pytesseract.image_to_string(preprocessed_image, lang='eng')
                    for line in text.splitlines():
                        cleaned = line.strip()
                        if cleaned:
                            elements.append({
                                'text': cleaned,
                                'font_size': 10,
                                'page': page_number,
                            })
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
        else:
            logger.info("Using pdfplumber for text extraction with header/footer filtering...")
            with pdfplumber.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf.pages, start=1):
                    logger.info(f"Extracting text from page {page_number}")
                    try:
                        page_height = page.height
                        header_footer_threshold = 0.1 * page_height
                        words = page.extract_words()
                        filtered_words = []
                        for w in words:
                            top = w['top']
                            bottom = w['bottom']
                            if top < header_footer_threshold or bottom > (page_height - header_footer_threshold):
                                continue
                            filtered_words.append(w)
                        lines_dict = {}
                        for w in filtered_words:
                            top_key = round(w['top'] / 3) * 3
                            lines_dict.setdefault(top_key, []).append(w['text'])
                        lines = [' '.join(lines_dict[k]) for k in sorted(lines_dict.keys())]
                        for line in lines:
                            cleaned = line.strip()
                            if cleaned:
                                elements.append({
                                    'text': cleaned,
                                    'font_size': 10,
                                    'page': page_number,
                                })
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_number}: {e}")
        logger.info(f"Extracted {len(elements)} text elements from PDF.")
        return elements

    def clean_and_merge_text(self, elements: List[Dict]) -> str:
        meaningful_lines = []
        url_pattern = re.compile(r'https?://\S+')
        boilerplate_patterns = [
            r'^\s*Page \d+\s*$',
            r'^\s*©.*$',
            r'^\s*All rights reserved.*$',
            r'^\s*Terms and Conditions.*$',
            r'^\s*Privacy Policy.*$',
            r'^\s*Home\s*$',
            r'^\s*Contact Us\s*$',
            r'^\s*About Us\s*$',
            r'^\s*Services\s*$',
            r'^\s*Get a Quote\s*$',
            r'^\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}',
            r'^\s*[-_=]{3,}\s*$',
        ]
        for el in elements:
            text = el['text'].strip()
            if len(text) < 15:
                continue
            if url_pattern.search(text):
                continue
            if any(re.match(pat, text, re.IGNORECASE) for pat in boilerplate_patterns):
                continue
            if re.fullmatch(r'[\W_]+', text):
                continue
            meaningful_lines.append(text)
        cleaned_text = '\n\n'.join(meaningful_lines)
        logger.info(f"Cleaned text length after filtering: {len(cleaned_text)}")
        logger.info(f"Cleaned text preview: {cleaned_text[:500]}")
        return cleaned_text

    def generate_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        if not text or len(text) < 30:
            logger.warning("Insufficient text for QA generation.")
            return []

        paragraphs = simple_paragraph_tokenize(text)
        logger.info(f"Total paragraphs extracted: {len(paragraphs)}")
        chunks = chunk_text_paragraphs(paragraphs, max_chars=1000)
        logger.info(f"Total chunks created: {len(chunks)}")
        for idx, chunk in enumerate(chunks):
            logger.info(f"Chunk {idx+1} preview: {chunk[:200]}")

        qa_pairs = []
        for chunk_idx, chunk in enumerate(chunks):
            try:
                prompt = (
                    "You are a professional AI assistant tasked with generating detailed question-answer pairs for chatbot training.\n\n"
                    "Instructions:\n"
                    "- Read the following text carefully.\n"
                    "- Generate 40-45 clear, relevant, and diverse question-answer pairs based on the text.\n"
                    "- Use first-person plural pronouns (we, us, our) in answers, as if you represent the company.\n"
                    "- Phrase questions from the perspective of a customer or user seeking information.\n"
                    "- Provide informative, concise answers that reflect the company’s voice and policies.\n"
                    "- Format each pair exactly as:\n"
                    "Q: [question]\n"
                    "A: [answer]\n"
                    "- Do not include any other text or explanation.\n\n"
                    f"Text:\n{chunk}\n\n"
                    "Please provide the Q&A pairs now."
                    )

                logger.info(f"Sending chunk {chunk_idx+1} to Ollama (model: {self.ollama_model}) for QnA generation.")
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    stream=False
                )
                raw_output = response['message']['content']
                logger.info(f"Ollama raw response for chunk {chunk_idx+1}:\n{raw_output}")

                lines = [l.strip() for l in raw_output.split('\n') if l.strip()]
                current_q = None
                for line in lines:
                    q_match = re.match(r'^Q:\s*(.*)', line)
                    a_match = re.match(r'^A:\s*(.*)', line)
                    if q_match:
                        current_q = q_match.group(1).strip()
                    elif a_match and current_q:
                        answer = a_match.group(1).strip()
                        # Basic validation: question contains '?' and reasonable length
                        if len(current_q) > 5 and len(answer) > 5 and '?' in current_q:
                            qa_pairs.append({'question': current_q, 'answer': answer})
                        current_q = None
            except Exception as e:
                logger.error(f"Ollama QnA generation failed on chunk {chunk_idx+1}: {e}")
                continue

        logger.info(f"Generated {len(qa_pairs)} Q&A pairs in total.")
        return qa_pairs

    def convert_to_json(self, pdf_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> Dict:
        logger.info(f"Starting PDF to JSON conversion for: {pdf_path}")
        elements = self.extract_text_with_formatting(pdf_path)

        pages = {}
        for el in elements:
            page_num = el['page']
            pages.setdefault(page_num, []).append(el['text'])

        pages_json = []
        for pnum in sorted(pages.keys()):
            full_text = ' '.join(pages[pnum])
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            pages_json.append({
                "page_number": pnum,
                "content": [
                    {
                        "type": "text",
                        "value": full_text
                    }
                ]
            })

        cleaned_text = self.clean_and_merge_text(elements)
        qa_pairs = self.generate_qa_pairs(cleaned_text)

        # Convert qa_pairs to Gemini training format
        gemini_qa_pairs = [
            {
                "prompt": qa["question"],
                "completion": qa["answer"]
            }
            for qa in qa_pairs
        ]

        result = {
            "pages": pages_json,
            "qa_pairs": gemini_qa_pairs,
        }

        if output_path:
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"JSON output written to {output_path}")
            except Exception as e:
                logger.error(f"Failed to write JSON output: {e}")

        logger.info("PDF to JSON conversion completed.")
        return result
