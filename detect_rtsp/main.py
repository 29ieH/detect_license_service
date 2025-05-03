import cv2
import time
import requests
import  websocket
import json
from dotenv import load_dotenv
import os
load_dotenv()  # Load từ file .env
rootFolder = os.getenv("root-folder")          # VD: /data/
vehicleLog = os.getenv("vehicle-log-folder")   # VD: vehicle_logs/
fullFolder = os.path.join(rootFolder, vehicleLog)
os.makedirs(fullFolder, exist_ok=True) 
ws = websocket.WebSocket()
ws.connect("ws://localhost:8080/ws/vehicle")
# ======== Cấu hình ========
RTSP_URL = "rtsp://admin:abcd1234@@192.168.4.8//Streaming/Channels/101/"
API_TOKEN = "da3bb68b831300f0930148c407e20f74e8a5c92d"
FRAME_INTERVAL = 1  # gửi 1 frame mỗi giây

# ======== Khởi tạo ========
headers = {
    'Authorization': f'Token {API_TOKEN}'
}
api_url = "https://api.platerecognizer.com/v1/plate-reader/"
# api_url_local = "http://localhost:5000/api/recognize"
# ======== Kết nối RTSP ========
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(" Không thể mở luồng RTSP.")
    exit()
print(" Đang kết nối luồng RTSP... Nhấn Ctrl+C để dừng.")
last_sent_time = 0
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Không đọc được frame.")
            break

        # Gửi mỗi FRAME_INTERVAL giây
        if time.time() - last_sent_time >= FRAME_INTERVAL:
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'upload': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            response = requests.post(api_url, headers=headers, files=files)
            # files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
            # response = requests.post(api_url_local, headers=headers, files=files)
            data = response.json()
            # Hiển thị kết quả
            if "results" in data and len(data["results"]) > 0:
                for plate in data["results"]:
                    plate_number = plate['plate'].upper()
                    print(f"📷 Biển số phát hiện: {plate_number}")
                    timestmap = int(time.time());
                    fileName = f"{plate_number}_{timestmap}.jpg";
                    filePath = os.path.join(fullFolder,fileName);
                    cv2.imwrite(filePath,frame)
                    print("path:: "+fullFolder+"\"+fileName)
                    message = {
                        "licensePlate":plate_number,
                        "imageUrl":f"{fullFolder}\{fileName}",
                        "entryDate": time.strftime('%Y-%m-%dT%H:%M:%S')
                    }
                    ws.send(json.dumps(message))
            else:
                print("🚫 Không phát hiện biển số.")

            last_sent_time = time.time()

        # (Optional) Hiển thị frame
        cv2.imshow('RTSP Stream', frame)
        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⏹ Dừng chương trình.")

# ======== Giải phóng ========
cap.release()
cv2.destroyAllWindows()
