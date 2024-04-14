import torch
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import cv2
import os 

Weight = "./model/best.pt"
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=Weight)

Model = "./model/model.h5"
defect_model = load_model(Model)

class_names = ['Defective', 'Normal', 'No-OBJ']

normal_path = "./saved_data/normal/"
defective_path = "./saved_data/defective/"

def get_file_count(folder_path):
    count = 0
    for files in os.listdir(folder_path):
        count += 1
    return count

def save_img(img, index):
    if index == 1: # consider normal image
        count = get_file_count(normal_path)
        fname = "n_img_"+str(count)+".jpeg"
        cv2.imwrite(normal_path+fname, img)
    else:
        count = get_file_count(defective_path)
        fname = "d_img_"+str(count)+".jpeg"
        cv2.imwrite(defective_path+fname, img)
    print("[INFO].. IMAGE SAVED ..")

def detection(img):
    print("[INFO] Detecting object")
    results = yolo_model(img)
    try:
        bboxes = results.pandas().xyxy[0]
        return bboxes, img
    except:
        print("[WARNING]: In image not able to put box on image")
        return None, img

def recognition(boxes, frame):
    index, conf = 2, 100 # Default Values to index and conf just to say no object
    cluster = []
    if boxes is not None:
        for i, box in boxes.iterrows():
            x_min, y_min, x_max, y_max = int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)
            frame = cv2.rectangle(frame, (x_min-5, y_min-20), (x_max+5, y_max+20), (0, 255, 0), 2)
            # print(i, x_min, y_min, x_max, y_max )
            crop_img = frame[y_min:y_max, x_min:x_max]
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
            crop_img = cv2.resize(crop_img, (640, 640))
            crop_img = np.expand_dims(crop_img, axis=0)
            predict_results =  defect_model.predict(crop_img)
            # print("PREDICTION", predict_results)
            score = tf.nn.softmax(predict_results[0])
            index = np.argmax(score)
            conf = 100 * np.max(score)
            frame = cv2.putText(frame, class_names[index], (x_min, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            print("[OBJECT IS :{0}] & [CONFIDENCE IS :{1}%]".format(class_names[index], int(conf)))
            # print("INDEX:",index, "CONF:",conf)
            cluster.append([index, int(conf)])
            save_img(frame, index)
        return cluster, frame
    else:
        print("[INFO].. NO OBJECT BOUNDRIES FOUND!")
        return cluster, frame

cam = cv2.VideoCapture(0)

def main():
    while True:
        ret, frame = cam.read()
        if ret:
            # frame = cv2.imread("./test_data/2.jpeg")
            frame = cv2.resize(frame, (200, 200))
            boxes, frame = detection(frame)
            cluster, frame = recognition(boxes, frame)
            print(cluster)
            frame = cv2.resize(frame, (200,200))
            cv2.imshow("OUT", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("[INFO].. No Camera Input Detected")

    cam.release()
    cv2.destroyAllWindows()


if __name__== "__main__":
    main()