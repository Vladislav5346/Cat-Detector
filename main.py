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

            r_norm = r / 255
            g_norm = g / 255
            b_norm = b / 255
            
            max_color = max(r_norm, g_norm, b_norm)
            min_color = min(r_norm, g_norm, b_norm)
            saturation = (max_color - min_color) / max_color if max_color > 0 else 0
            brightness = (r_norm + g_norm + b_norm) / 3
            
            if brightness > 0.7:
                color = "white"
            elif brightness < 0.2:
                color = "black"
            elif saturation < 0.2:
                color = "gray"
            elif r_norm > g_norm and r_norm > b_norm and r_norm > 0.4:
                color = "orange"
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
