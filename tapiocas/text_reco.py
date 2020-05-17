from enum import Enum
import pytesseract
import logging


# OCR ENGINE MODE: https://tesseract.patagames.com/help/html/T_Patagames_Ocr_Enums_OcrEngineMode.htm
class OEM(Enum):
    TESSERACT_ONLY = 0  # Run Tesseract only - fastest
    CUBE_ONLY = 1  # Run Cube only - better accuracy, but slower
    TESSERACT_CUBE_COMBINED = 2  # Run both and combine results - best accuracy
    DEFAULT = 3  # automatic


# PAGE SEG MODE: https://tesseract.patagames.com/help/html/T_Patagames_Ocr_Enums_PageSegMode.htm
class PSM(Enum):
    OSD_ONLY = 0  # Orientation and script detection only
    AUTO_OSD = 1  # Automatic page segmentation with orientation and script detection
    AUTO_ONLY = 2  # Automatic page segmentation, but no OSD, or OCR
    AUTO = 3  # Fully automatic page segmentation, but no OSD
    SINGLE_COLUMN = 4  # Assume a single column of text of variable sizes
    SINGLE_BLOCK_VERT_TEXT = 5  # Assume a single uniform block of vertically aligned text
    SINGLE_BLOCK = 6  # Assume a single uniform block of text
    SINGLE_LINE = 7  # Treat the image as a single text line
    SINGLE_WORD = 8  # Treat the image as a single word
    CIRCLE_WORD = 9  # Treat the image as a single word in a circle
    SINGLE_CHAR = 10  # Treat the image as a single character
    SPARSE_TEXT = 11  # Find as much text as possible in no particular order
    SPARSE_TEXT_OSD = 12  # Sparse text with orientation and script det
    RAW_LINE = 13  # Treat the image as a single text line, bypassing hacks that are Tesseract-specific


def find_text(image, oem=OEM.DEFAULT, psm=PSM.SPARSE_TEXT, whitelist=None):
    w_str = f" -c tessedit_char_whitelist={whitelist}" if whitelist else ""
    config = f"--oem {oem.value} --psm {psm.value}{w_str}"
    logging.debug(f"looking for text with tesseract config: '{config}'")
    s = ""
    try:
        s = pytesseract.image_to_string(image, config=config)
    except Exception as e:
        logging.exception(e)
    return s
