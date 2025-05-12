# Step By Step JetsonNano
Setelah menguji model di Colab dan memastikan semuanya berfungsi:

1. Download model ONNX yang dihasilkan
2. Transfer ke Jetson Nano
3. Instal dependensi di Jetson Nano:

sudo apt-get update
sudo apt-get install python3-pip
pip3 install onnxruntime opencv-python numpy

4. Jalankan Kode khusus Jetson Nano di IDE Local, bagian:
# ---- 8. Kode untuk Jetson Nano (akan dijalankan nanti di Jetson) ----

"""
Di bawah ini adalah kode yang akan dijalankan di Jetson Nano
setelah pengujian di Colab selesai:

# Di Jetson Nano:
import cv2
import time
import numpy as np
import onnxruntime as ort

# Load ONNX model
ort_session = ort.InferenceSession("mobilenet_v2.onnx")

# Setup kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load labels
with open("imagenet_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Preprocessing functions
def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32)
    img /= 255.0
    img = (img - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]
    return img[None, :]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocessing
    input_data = preprocess(frame)
    
    # Inference
    start = time.time()
    outputs = ort_session.run(None, {'input': input_data})
    inference_time = (time.time() - start) * 1000
    
    # Get prediction
    scores = outputs[0][0]
    idx = np.argmax(scores)
    prob = np.exp(scores[idx]) / np.sum(np.exp(scores))
    label = labels[idx]
    
    # Display hasil
    cv2.putText(frame, f"{label}: {prob:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Inference: {inference_time:.1f}ms", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Jetson Nano CV Demo", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""
