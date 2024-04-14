import cv2
import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# load model path's
Weight = "./model/best.pt"
Model = "./model/model.h5"

def detection(img):
    print("[INFO] Detecting object")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=Weight)
    results = yolo_model(img)
    try:
        bboxes = results.pandas().xyxy[0]
        return bboxes, img
    except:
        print("[WARNING]: In image not able to put box on image")
        return None, img

def Recognition(img, boxes):
    defect_model = load_model(Model)
    # try:
    for i, box in boxes.iterrows():
        x_min, y_min, x_max, y_max = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
        detected_object = img.crop((x_min, y_min, x_max, y_max))
        detected_object = cv2.cvtColor(np.array(detected_object), cv2.COLOR_RGB2BGR)
        detected_object = cv2.resize(detected_object, (640, 640))
        detected_object = np.expand_dims(detected_object, axis=0)

        prediction = defect_model.predict(detected_object)
        score = tf.nn.softmax(prediction[0])
        class_names = ['Defective', 'Normal']
        prediction_label = "{} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
    
    return prediction_label, img
    # except:
    #     print("[WARNING] Unable to predict")
    #     return None, img
    

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open camera exiting from the exicution!")
        exit()
    
    while True:
        ret, frame = cap.read()
        # frame = cv2.imread("./test_data/1.jpeg")
        cv2.imshow('Original', frame)
        if not ret:
            print("Error: Couldn't capture image frame")
            break
        # img = np.zeros((480, 640, 3), dtype=np.uint8)
        boxes, frame = detection(frame)
        print("THIS IS BBOX:", boxes)
        if boxes is not None:
            print("[INFO] Sending for recognition")
            result, frame = Recognition(frame, boxes)
            print("[OBJECT IS] : ", result)
        else:
            print("[WARNING] unable to recognise image")
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
