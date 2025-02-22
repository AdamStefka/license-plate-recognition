from collections import defaultdict

import easyocr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("../notebooks/runs/detect/train/weights/best.pt")

reader = easyocr.Reader(['en'])

def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        print(f"bbox: {bbox}, text: {text}, score: {score}")
        text = text.upper().replace(' ', '').replace('_', '').replace('-', '').replace('.', '').replace(';', '').replace(',', '').replace(':', '').replace('?', '').replace('!', '').replace('*', '').replace('(', '').replace(')', '').replace("'", "").replace('"', '').replace('#', '').replace('[', '').replace(']', '')
        return text, score

    return None, None


def areaFilter(minArea, inputImage):
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage

def get_plate(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xplate1, yplate1, xplate2, yplate2, car_id = vehicle_track_ids[j]

        if x1 > xplate1 and y1 > yplate1 and x2 < xplate2 and y2 < yplate2:
            car_idx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_idx]

    return -1, -1, -1, -1, -1


def write_csv(results, output_path):
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{}\n'.format('frame_nmr', 'license_plate_bbox',
                                          'license_plate_bbox_score', 'license_number', 'license_number_score'))

        for frame_nmr in results.keys():
            for lp in results[frame_nmr].keys():
                print(results[frame_nmr])
                if 'text' in results[frame_nmr][lp].keys():
                    f.write('{},{},{},{},{}\n'.format(frame_nmr,'[{} {} {} {}]'.format(
                        results[frame_nmr][lp]['bbox'][0],
                        results[frame_nmr][lp]['bbox'][1],
                        results[frame_nmr][lp]['bbox'][2],
                        results[frame_nmr][lp]['bbox'][3]),
                                                      results[frame_nmr][lp]['bbox_score'],
                                                      results[frame_nmr][lp]['text'],
                                                      results[frame_nmr][lp]['text_score'])
                            )
        f.close()