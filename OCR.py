# Used OCR to extract Bengali text from a PDF document

!apt-get install -y poppler-utils
!pip install pdf2image pytesseract
!apt install tesseract-ocr -y
!apt install tesseract-ocr-ben -y

from pdf2image import convert_from_path
import pytesseract
import os
from IPython.display import display
from PIL import Image

pdf_path = "HSC26-Bangla1st-Paper.pdf"  # Replace with your actual file name

# Convert PDF to list of images
images = convert_from_path(pdf_path)
print(f"Total pages: {len(images)}")

bengali_text = ""

for i, img in enumerate(images):
    print(f"🔍 Processing page {i+1}...")
    text = pytesseract.image_to_string(img, lang='ben')  # Bengali language
    bengali_text += f"\n\n--- Page {i+1} ---\n{text}"

print("✅ OCR complete!")

with open("extracted_bengali_text.txt", "w", encoding='utf-8') as f:
    f.write(bengali_text)