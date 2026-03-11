'''Программа для обнаружения кошек на изображении и определения их цвета (рыжий, черный, белый или смешанный) с помощью opencv и модели YOLOv8.'''

import cv2, sys, os, numpy
from ultralytics import YOLO


def detect_cats(img_path: str):
    if not os.path.exists("models"):
        os.makedirs("models")

    model = YOLO("models/yolov8n.pt")

    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    if max(height, width) > 1024:
        scale = 1024 / max(height, width)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))

    results = model(img)

    for r in results:
        for box in r.boxes:

            class_id = int(box.cls[0])

            if class_id != 15:  # 15 = cat
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cat = img[y1:y2, x1:x2]

            avg_color = numpy.mean(cat, axis=(0,1))
            b, g, r = avg_color

            if r > 150 and g > 120 and b < 100:
                color = "orange"
            elif r < 100 and g < 100 and b < 100:
                color = "black"
            elif r > 200 and g > 200 and b > 200:
                color = "white"
            else:
                color = "mixed"

            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(img,f"Cat: {color}",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Cats detector", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        detect_cats(sys.argv[1])
    else:
        detect_cats(input("Введите путь к изображению: "))
