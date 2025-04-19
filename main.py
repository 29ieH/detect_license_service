import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ExifTags
import io
import base64
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging
from datetime import datetime
import re

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Cấu hình Roboflow API
ROBOFLOW_API_KEY = "TV7JZBQXSn4Dmd6Ena7L"
DETECT_LICENSE_ENDPOINT = "https://serverless.roboflow.com/infer/workflows/thaihien/detect-license"
DETECT_CHAR_ENDPOINT = "https://serverless.roboflow.com/infer/workflows/thaihien/detect-char"

# Thư mục lưu ảnh đã cắt
CROPPED_PLATES_DIR = "cropped_plates"
CHAR_PLATES_DIR = "char_plates"
if not os.path.exists(CROPPED_PLATES_DIR):
    os.makedirs(CROPPED_PLATES_DIR)
if not os.path.exists(CHAR_PLATES_DIR):
    os.makedirs(CHAR_PLATES_DIR)
# Tạo session với retry
def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = create_session()
# Tiền xử lý 
def preprocess_plate_image(image):
    try:
        # Convert sang grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
         # Tăng tương phản
        gray = cv2.equalizeHist(gray)
         # Làm mượt và loại bỏ nhiễu nhỏ
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Dùng adaptive threshold (nếu muốn nhấn mạnh ký tự trắng trên nền đen)
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
           # Chuyển lại thành ảnh 3 kênh nếu model yêu cầu ảnh màu
        processed = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        return processed;
    except Exception as e:
        logging.error(f"Lỗi tiền xử lý ảnh: {e}")
        return image
# Hàm sửa hướng xoay ảnh
def correct_image_orientation(image):
    try:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = pil_image._getexif()
        if exif is not None and orientation in exif:
            if exif[orientation] == 3:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif exif[orientation] == 6:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif exif[orientation] == 8:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        logging.info("Đã sửa hướng ảnh thành công")
    except Exception as e:
        logging.error(f"Không thể sửa hướng ảnh: {e}")
    return image

# Hàm chuyển ảnh thành base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

# Hàm gọi API Workflow để nhận diện khung biển số
def call_detect_license(image_base64):
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": image_base64}
        }
    }
    headers = {"Content-Type": "application/json"}
    # logging.info(f"Gửi payload đến detect-license: {payload}")
    try:
        response = session.post(DETECT_LICENSE_ENDPOINT, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        logging.info(f"Kết quả từ detect-license: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        raise Exception(f"HTTP Error: {e.response.status_code} - {e.response.text}")

# Hàm gọi API Workflow để nhận diện ký tự
def call_detect_char(image_base64):
    payload = {
        "api_key": ROBOFLOW_API_KEY,
        "inputs": {
            "image": {"type": "base64", "value": image_base64}
        }
    }
    headers = {"Content-Type": "application/json"}
    # logging.info(f"Gửi payload đến detect-char: {payload}")
    try:
        response = session.post(DETECT_CHAR_ENDPOINT, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        logging.info(f"Kết quả từ detect-char: {result}")
        return result
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        raise Exception(f"HTTP Error: {e.response.status_code} - {e.response.text}")

# Hàm cắt khung biển số và lưu ảnh
def crop_license_plate(image, predictions):
    if not predictions:
        logging.warning("Không có predictions để cắt khung biển số")
        return None
    for pred in predictions:
        logging.info(f"Class: {pred['class']}, Confidence: {pred['confidence']}")
    for pred in predictions:
        x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        cropped = image[y1:y2, x1:x2]
        # Lưu ảnh đã cắt
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cropped_path = os.path.join(CROPPED_PLATES_DIR, f"plate_{timestamp}.jpg")
        cv2.imwrite(cropped_path, cropped)
        logging.info(f"Đã lưu ảnh cắt tại: {cropped_path}")
        logging.info(f"Đã cắt khung biển số: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return cropped
    return None

# Hàm nhận diện ký tự từ ảnh đã cắt
# def recognize_characters(cropped_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    char_path = os.path.join(CHAR_PLATES_DIR, f"char_{timestamp}.jpg")
    cv2.imwrite(char_path,cropped_image)
    image_base64 = image_to_base64(cropped_image)
    prediction = call_detect_char(image_base64)

    try:
        predictions = prediction['outputs'][0]['predictions']['predictions']
    except (KeyError, IndexError):
        raise Exception("Không nhận diện được ký tự")

    if not predictions:
        raise Exception("Không có ký tự nào được nhận diện")

    # Lấy thông tin x, y, class, height
    chars = [(pred['x'], pred['y'], pred['class'], pred['height']) for pred in predictions 
             if (
                  (pred['class'] == '1' and pred['confidence'] > 0.3)
                 or (pred['class'] != '1' and pred['confidence'] > 0.65) 
             )]

    if not chars:
        raise Exception("Không có ký tự nào đạt ngưỡng độ tin cậy")

    # Sắp xếp các ký tự theo y để nhóm thành hàng
    chars.sort(key=lambda c: c[1])

    # Tìm khoảng cách y lớn nhất để tách hàng (nếu có)
    lines = []
    if len(chars) > 1:
        # Tính khoảng cách y giữa các ký tự liên tiếp
        y_diffs = [(chars[i][1] - chars[i-1][1], i) for i in range(1, len(chars))]
        if y_diffs:
            max_y_diff, split_idx = max(y_diffs, key=lambda x: x[0])
            # Nếu khoảng cách y lớn nhất đủ lớn (ví dụ: > 5), tách thành 2 hàng
            if max_y_diff > 5:
                lines.append(chars[:split_idx])
                lines.append(chars[split_idx:])
            else:
                lines.append(chars)  # Chỉ có 1 hàng
        else:
            lines.append(chars)
    else:
        lines.append(chars)

    # Sort từng dòng theo x (từ trái qua phải)
    for line in lines:
        line.sort(key=lambda c: c[0])

    # Sort các dòng theo y (từ trên xuống dưới)
    lines.sort(key=lambda line: line[0][1])

    # Tạo chuỗi kết quả
    plate_number_parts = [''.join([label for _, _, label, _ in line]) for line in lines]
    plate_number = ' '.join(plate_number_parts)

    # Nếu chỉ có một hàng (biển số xe ô tô), thêm dấu cách giữa phần chữ và số
    if len(plate_number_parts) == 1:
        part = plate_number_parts[0]
        # Tìm vị trí sau ký tự chữ (thường là sau ký tự thứ 3, ví dụ: "18A")
        split_pos = 3
        for i in range(len(part)):
            if part[i].isalpha():
                split_pos = i + 1
                break
        plate_number = part[:split_pos] + ' ' + part[split_pos:]

    return plate_number
def recognize_characters(cropped_image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    char_path = os.path.join(CHAR_PLATES_DIR, f"plate_{timestamp}.jpg")
    cv2.imwrite(char_path, cropped_image)
    image_base64 = image_to_base64(cropped_image)
    prediction = call_detect_char(image_base64)

    try:
        predictions = prediction['outputs'][0]['predictions']['predictions']
    except (KeyError, IndexError):
        raise Exception("Không nhận diện được ký tự")

    if not predictions:
        raise Exception("Không có ký tự nào được nhận diện")

    # Lấy thông tin x, y, class, height
    chars = [(pred['x'], pred['y'], pred['class'], pred['height']) for pred in predictions 
             if (
                 (pred['class'] == 1 and pred['confidence'] > 0.3) or
                 (pred['class'] != 1 and pred['confidence'] > 0.5)
             )]

    if not chars:
        raise Exception("Không có ký tự nào đạt ngưỡng độ tin cậy")

    # Sắp xếp các ký tự theo y để nhóm thành hàng
    chars.sort(key=lambda c: c[1])

    # Tìm khoảng cách y lớn nhất để tách hàng (nếu có)
    lines = []
    if len(chars) > 1:
        y_diffs = [(chars[i][1] - chars[i-1][1], i) for i in range(1, len(chars))]
        if y_diffs:
            max_y_diff, split_idx = max(y_diffs, key=lambda x: x[0])
            if max_y_diff > 5:
                lines.append(chars[:split_idx])
                lines.append(chars[split_idx:])
            else:
                lines.append(chars)
        else:
            lines.append(chars)
    else:
        lines.append(chars)

    # Sort từng dòng theo x (từ trái qua phải)
    for line in lines:
        line.sort(key=lambda c: c[0])

    # Sort các dòng theo y (từ trên xuống dưới)
    lines.sort(key=lambda line: line[0][1])

    # Tạo chuỗi kết quả
    plate_number_parts = [''.join([label for _, _, label, _ in line]) for line in lines]

    # Hàm sửa lỗi nhận diện ký tự
    def correct_char(char, should_be_letter):
        if should_be_letter:
            # Nếu cần là chữ cái nhưng lại là số (như "8"), sửa thành chữ (như "B")
            if char.isdigit():
                if char == "8":
                    return "B"
                elif char == "5":
                    return "S"
                elif char == "0":
                    return "O"
                return char  # Nếu không biết sửa, giữ nguyên
            return char
        else:
            # Nếu cần là số nhưng lại là chữ (như "B"), sửa thành số (như "8")
            if char.isalpha():
                if char == "B":
                    return "8"
                elif char == "S":
                    return "5"
                elif char == "O":
                    return "0"
                return char  # Nếu không biết sửa, giữ nguyên
            return char

    # Xử lý tùy theo loại biển số
    if len(plate_number_parts) == 2:  # Biển số xe máy (2 hàng)
        top_part, bottom_part = plate_number_parts

        # Định dạng mong muốn: hàng trên "[2 số][1 chữ cái][1 số]", hàng dưới "[5 số]"
        top_pattern = r"^[0-9]{2}[A-Z][0-9]$"
        bottom_pattern = r"^[0-9]{4,5}$"

        # Sửa hàng trên
        if len(top_part) == 4:  # Đúng độ dài
            corrected_top = []
            for i, char in enumerate(top_part):
                if i in [0, 1, 3]:  # Vị trí 0, 1, 3 phải là số
                    corrected_top.append(correct_char(char, should_be_letter=False))
                else:  # Vị trí 2 phải là chữ cái
                    corrected_top.append(correct_char(char, should_be_letter=True))
            top_part = ''.join(corrected_top)

        # Sửa hàng dưới
        if len(bottom_part) == 5:  # Đúng độ dài
            corrected_bottom = [correct_char(char, should_be_letter=False) for char in bottom_part]
            bottom_part = ''.join(corrected_bottom)

        # Kiểm tra định dạng bằng regex
        if not re.match(top_pattern, top_part):
            raise Exception(f"Hàng trên không đúng định dạng: {top_part}")
        if not re.match(bottom_pattern, bottom_part):
            raise Exception(f"Hàng dưới không đúng định dạng: {bottom_part}")
        plate_number = f"{top_part} {bottom_part}"

    else:  # Biển số xe ô tô (1 hàng)
        part = plate_number_parts[0]
        # Định dạng mong muốn: "[2 số][1 chữ cái][5 số]"
        pattern = r"^[0-9]{2}[A-Z][0-9]{5}$"

        if len(part) == 8:  # Đúng độ dài
            corrected_part = []
            for i, char in enumerate(part):
                if i == 2:  # Vị trí 2 phải là chữ cái
                    corrected_part.append(correct_char(char, should_be_letter=True))
                else:  # Các vị trí khác phải là số
                    corrected_part.append(correct_char(char, should_be_letter=False))
            part = ''.join(corrected_part)

        # Kiểm tra định dạng
        if not re.match(pattern, part):
            raise Exception(f"Biển số xe ô tô không đúng định dạng: {part}")

        # Thêm dấu cách giữa phần chữ và số
        plate_number = part[:3] + ' ' + part[3:]

    return plate_number

# Route API xử lý ảnh
@app.route('/api/recognize', methods=['POST'])
def recognize_plate():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file được upload'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file'}), 400
    try:
        image = Image.open(file.stream)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        # image_cv = correct_image_orientation(image_cv)
        image_base64 = image_to_base64(image_cv)
        plate_prediction = call_detect_license(image_base64)
        raw_output = plate_prediction.get("outputs", [])
        if not raw_output or not raw_output[0]["predictions"]["predictions"]:
            return jsonify({'error': 'Không phát hiện khung biển số'}), 400
        predictions = raw_output[0]["predictions"]["predictions"]
        cropped_plate = crop_license_plate(image_cv, predictions)
        if cropped_plate is None:
            return jsonify({'error': 'Không thể cắt khung biển số'}), 400
        plate_number = recognize_characters(cropped_plate)
        return jsonify({'plate_number': plate_number})
    
    except Exception as e:
        logging.error(f"Lỗi trong quá trình xử lý: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)