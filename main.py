import cv2
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")  # load a pretained model
results = model.predict(source='C:/Users/HP.Z.BOOK G3/Downloads/4804794-uhd_3840_2160_25fps.mp4',save=True, imgsz=640,conf=0.5)

cap = cv2.VideoCapture('pose.mp4')
while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (640, 640))
    if not ret:
        break
    results = model.predict(frame, save=True)
    print('results', results)
    detection = results[0].plot()
    cv2.imshow('YOLOv8 Pose Detection', detection)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()






