from collections import defaultdict

from ultralytics import YOLO
import cv2
from functions import read_license_plate, write_csv, areaFilter
from sort.sort import *
import numpy as np

model = YOLO("../notebooks/runs/detect/train/weights/best.pt")

cam = cv2.VideoCapture(0)

BINARY_THRESH = [100, 120, 150, 170, 190]
LAST_PARAM = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]

results = {}
mot_tracker = Sort()

frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cam.read()
    if ret:
        results[frame_nmr] = {}
        detections = model(frame, conf=0.4)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():

            prediction_dict = defaultdict(lambda: [set(), 0, []])

            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) == 0:
                detections_.append([x1, y1, x2, y2, score])

            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)


            for last_param in LAST_PARAM:
                for thresh in BINARY_THRESH:
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, thresh, 255, last_param)


                    minArea = 100
                    binaryImage = areaFilter(minArea, license_plate_crop_thresh)


                    kernelSize = 3
                    opIterations = 2
                    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
                    binaryImage = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE, morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)


                    license_plate_text, score_plate = read_license_plate(binaryImage)
                    if license_plate_text is None:
                        continue
                    prediction_dict[license_plate_text][0] = license_plate_text
                    prediction_dict[license_plate_text][1] += 1
                    prediction_dict[license_plate_text][2].append(score_plate)

            max_set = None
            max_core = None
            for key in prediction_dict:
                prediction_dict[key][2] = sum(prediction_dict[key][2]) / len(prediction_dict[key][2])
            if prediction_dict:
                max_key = max(prediction_dict, key=lambda k: prediction_dict[k][1])
                max_set = prediction_dict[max_key][0]
                max_score = prediction_dict[max_key][2]
            if max_set is not None:
                results[frame_nmr] = {'license_plate': {'bbox': [x1, y1, x2, y2],
                                                        'text': max_set,
                                                        'bbox_score': score,
                                                        'text_score': max_score}}

        detections_np = np.array(detections_) if len(detections_) > 0 else np.empty((0, 5))
        if detections_np.shape[0] > 0:
            track_ids = mot_tracker.update(detections_np)
        else:
            track_ids = []

        for obj in track_ids:
            x1, y1, x2, y2, obj_id = obj.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
        cv2.imshow("License Plate Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


write_csv(results, './test.csv')


cam.release()
cv2.destroyAllWindows()