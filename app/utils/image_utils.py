import cv2
import easyocr
import numpy as np


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(resized, -1, kernel)
    denoised = cv2.fastNlMeansDenoising(sharpened, None, 30, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_gold(preprocessed_image):
    reader = easyocr.Reader(['en'], gpu=False)
    reader.lang_char = '0123456789()'
    result = reader.readtext(preprocessed_image)
    text = ' '.join([item[1] for item in result])
    return text

def parse_gold(gold_text):
    import re
    cleaned_text = re.sub(r'[^0-9() ]', '', gold_text)
    gold_list = cleaned_text.split()
    gold_data = []
    for gold in gold_list:
        parts = gold.split('(')
        if len(parts) == 2:
            current_gold = parts[0].strip()
            total_gold = parts[1].strip(')')
            gold_data.append({"Gold_inventory": current_gold, "Gold_total": total_gold})
    return gold_data
