import pdfplumber
import re

def extract_text_by_page(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({
                    "page_number": i + 1,
                    "text": text.strip()
                })
    return pages



def is_heading(line):
    line = line.strip()
    if not line:
        return False
    if len(line.split()) > 10:  # headings are short
        return False
    if line.endswith('.'):  # headings don't end with period
        return False
    if line.isupper() and len(line) > 3:  # ALL CAPS = heading
        return True
    return False

def chunk_by_sections(pages):
    chunks = []
    current_heading = "INTRODUCTION"
    current_text = []
    current_pages = []

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            if is_heading(line):
                if current_text:
                    chunks.append({
                        "section": current_heading,
                        "text": " ".join(current_text),
                        "page_number": current_pages[0] if current_pages else 0
                    })
                current_heading = line.strip()
                current_text = []
                current_pages = []
            else:
                current_text.append(line.strip())
                current_pages.append(page["page_number"])

    if current_text:
        chunks.append({
            "section": current_heading,
            "text": " ".join(current_text),
            "page_number": current_pages[0] if current_pages else 0
        })

    return chunks

