# test_ocr.py
import pytesseract
# 确保这一行指向你刚才安装的路径
pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

print("Tesseract Version:", pytesseract.get_tesseract_version())
print("安装成功！")