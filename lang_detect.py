import os
import fitz  # PyMuPDF for PDF handling
import pandas as pd
import re
import pytesseract
from langdetect import detect, DetectorFactory
from collections import defaultdict
from PIL import Image  # For handling image data
from multiprocessing import Pool
from datetime import datetime
 
# Disable debug output
from icecream import ic
ic.disable()
 
# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Update Tesseract path
DetectorFactory.seed = 0
 
# Function to extract text using Tesseract OCR from image-based PDFs
def extract_text_from_pdf(pdf_path):
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} does not exist.")
   
    pdf_document = fitz.open(pdf_path)
    text_chunks = []
    is_scanned = False
 
    for page_num, page in enumerate(pdf_document):
        # Try to extract text directly
        text = page.get_text("text")
 
        if not text:  # If no text, it's likely scanned
            is_scanned = True
            # Use OCR for scanned pages
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
 
        text_cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text_chunks.append(text_cleaned)
 
    pdf_document.close()
    return text_chunks, is_scanned
 
# Function to detect languages and compute percentages
def detect_languages(text_chunks):
    language_counts = defaultdict(int)
    for text_chunk in text_chunks:
        try:
            detected_lang = detect(text_chunk)
            language_counts[detected_lang] += 1
        except:
            continue
 
    total_chunks = len(text_chunks)
    language_percentages = {lang: (count / total_chunks) * 100 for lang, count in language_counts.items()}
    dominant_language = max(language_percentages, key=language_percentages.get) if language_percentages else None
 
    return dominant_language, language_percentages
 
# Worker function to process a single PDF
def process_single_pdf(pdf_info):
    index, filename, pdf_path = pdf_info
    try:
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File {pdf_path} does not exist.")
       
        print(f"Processing Document {index + 1}: {pdf_path}")
 
        # Extract text from the PDF and determine if it's scanned
        text_chunks, is_scanned = extract_text_from_pdf(pdf_path)
 
        # Detect languages and calculate percentages
        dominant_language, language_percentages = detect_languages(text_chunks)
 
        # Prepare result with separate columns for each language
        result = {
            "Filename": filename,
            "Document Number": index + 1,
            "Dominant Language": dominant_language,
            "Is Scanned": is_scanned,
        }
        result.update(language_percentages)  # Add language percentages as separate columns
 
        return result
 
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred for Document {index + 1}: {e}")
    return None
 
# Function to save results to an Excel file
def save_to_excel(results, output_dir, file_name='language_distribution_results.xlsx'):
    os.makedirs(output_dir, exist_ok=True)
    excel_path = os.path.join(output_dir, file_name)
   
    # Normalize results to ensure consistent columns for all languages
    df = pd.json_normalize(results).fillna(0)  # Fill missing languages with 0%
    if os.path.exists(excel_path):
        # Append to existing file
        df_existing = pd.read_excel(excel_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_excel(excel_path, index=False)
    else:
        # Create new file
        df.to_excel(excel_path, index=False)
    print(f"Results saved to {excel_path}")
 
# Main function to process all PDFs and save results
def process_pdfs(input_folder, csv_file, output_dir, num_processes=4):
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
 
    # Read the CSV file with filenames
    filenames_df = pd.read_csv(csv_file)
    pdf_infos = [(index, row['filename'], os.path.join(input_folder, f"{row['filename']}.pdf")) for index, row in filenames_df.iterrows()]
 
    results = []
    # Process PDFs in parallel
    with Pool(num_processes) as pool:
        for result in pool.imap_unordered(process_single_pdf, pdf_infos):
            if result:
                results.append(result)
                # Save to Excel incrementally for real-time progress
                save_to_excel([result], output_dir)
 
    # Save final results in case all are needed in one file
    save_to_excel(results, output_dir, file_name='final_language_distribution_results.xlsx')
 
# Define paths and run
input_folder = "/FE_Documents/ISIN/"
csv_file = "/home/dinesh/vs_works/filename.csv"  # CSV containing filenames
output_dir = '/home/dinesh/vs_works/lang_server_output'  # Output directory for Excel
num_processes = 4  # Number of parallel processes
 
if __name__ == "__main__":
    process_pdfs(input_folder, csv_file, output_dir, num_processes)
 