from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import cv2
import torch

np.set_printoptions(suppress=True)

Weight = "./model/best.pt"
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=Weight)

model = load_model("./mk_models/keras_models/keras_model.h5", compile=False)
class_names = open("./mk_models/keras_models/labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


THR_L = 140 # ADJUST THE PADDING OVER DETECTION
THR_H = 35

def do_prediction(img):
    # img = cv2.resize(img, (640, 640))
    results = yolo_model(img)
    bboxes = results.pandas().xyxy[0]
    # print(bboxes)
    if bboxes is not None:
        for i, box in bboxes.iterrows():
            x_min, y_min, x_max, y_max = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
            crop_img = img[y_min-THR_H:y_max+THR_H, x_min-THR_L:x_max+THR_L]
            img = cv2.rectangle(img, (x_min-THR_L, y_min-THR_H), (x_max+THR_L, y_max+THR_H), (0, 255, 0), 2) # adjusted padding to get the complete object
            # image = Image.open(crop_img).convert("RGB")
            image  = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            image_array = np.asarray(image)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data[0] = normalized_image_array
            prediction = model.predict(data)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = int(prediction[0][index]*100)
            # Print prediction and confidence score
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", confidence_score)
            text = str(class_name[2:])+" : "+str(confidence_score)+"%"
            img = cv2.putText(img, text, (x_min-30, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.imshow("OUT", img)



cap = cv2.VideoCapture("./videos/normal/normal4.mp4")

def main():
    if not cap.isOpened():
        print("Error: Unable to open video")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('INPUT', frame)
        do_prediction(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()