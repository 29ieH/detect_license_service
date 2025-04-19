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

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Cấu hình Roboflow API
ROBOFLOW_API_KEY = "TV7JZBQXSn4Dmd6Ena7L"
DETECT_LICENSE_ENDPOINT = "https://serverless.roboflow.com/infer/workflows/thaihien/detect-license"
DETECT_CHAR_ENDPOINT = "https://serverless.roboflow.com/infer/workflows/thaihien/detect-char"

# Thư mục lưu ảnh đã cắt
CROPPED_PLATES_DIR = "cropped_plates"
if not os.path.exists(CROPPED_PLATES_DIR):
    os.makedirs(CROPPED_PLATES_DIR)

# Tạo session với retry
def create_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

session = create_session()

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
    logging.info(f"Gửi payload đến detect-license: {payload}")
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
    logging.info(f"Gửi payload đến detect-char: {payload}")
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
def recognize_characters(cropped_image):
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
        (pred['class'] == 1 and pred['confidence'] > 0.20) or
        (pred['class'] != 1 and pred['confidence'] > 0.45)
    )
    ]
    # Nhóm theo y/height để xử lý nghiêng
    lines = []
    tolerance = 0.4
    for char in chars:
        x, y, label, height = char
        ratio = y / height
        placed = False
        for line in lines:
            ref_ratio = line[0][1] / line[0][3]
            if abs(ref_ratio - ratio) < tolerance:
                line.append(char)
                placed = True
                break
        if not placed:
            lines.append([char])

    # Sort từng dòng theo x, sau đó sort dòng theo y
    for line in lines:
        line.sort(key=lambda c: c[0])
    lines.sort(key=lambda line: line[0][1])

    plate_number = ''.join([label for line in lines for _, _, label, _ in line])
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
        image_cv = correct_image_orientation(image_cv)
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