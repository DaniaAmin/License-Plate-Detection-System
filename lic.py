from ultralytics import YOLO
# model=YOLO('license_plate_detector.pt');
# predict=model.predict('download.jpg',save=True);
# print(predict);
import numpy as np
from ultralytics import YOLO
import cv2;
from sort.sort import *
from util import get_car,read_license_plate,write_csv
import util
#test vedio soruce
#https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/


#resutls
results={};
#Tracking object.
mot_tracker=Sort();
#load model
coco_model=YOLO('yolov8n.pt');
license_plate_detector=YOLO("license_plate_detector.pt");

#load vedio
cap=cv2.VideoCapture('pexels_videos_2103099 (540p).mp4');
import cv2
import numpy as np

def resize_frame(frame):
  """Resizes a frame to half its original size."""
  width, height, _ = frame.shape
  new_width = int(width / 2)
  new_height = int(height / 2)
  return cv2.resize(frame, (new_width, new_height))





#object classes
vehicles=[2,3,5,7];
#read frames
frame_nmr=-1;
ret=True;
while ret:
    frame_nmr+=1;
    ret,frame=cap.read();
    #detect vehicles
    if ret and frame_nmr<30:
        results[frame_nmr]={};
        resized_frame = resize_frame(frame)
        detections = coco_model(frame)[0]
        detections_=[];
        for detection in detections.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=detection;
            if int(class_id) in vehicles:
                detections_.append([x1,y1,x2,y2,score]);

        #track vehicels
        #track_ids store information about detected object and add extra id columns of objects.
        track_ids=mot_tracker.update(np.asarray(detections_));

        #license plate detection
        license_plates = license_plate_detector(frame)[0]
        print(detections);
        for license_plate in license_plates.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id=license_plate;
            #Assign license plates to car
            xcar1,ycar1,xcar2,ycar2,car_id=get_car(license_plate,track_ids);
            #crop license plate from vedio
            #we perform all this process for easyocr to read and recognize faster.
            if car_id!=-1:
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV);

                # cv2.imshow("Original_crop",license_plate_crop)
                # cv2.imshow("license_plate_threshold", license_plate_crop_thresh);
                # cv2.waitKey(0);



                # Read license plate numbers
                license_plate_text, license_plat_text_score = read_license_plate(license_plate_crop_thresh);
                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score ,
                                                                    'text_score': license_plat_text_score}}
    # cv2.imshow("Model",frame);

#write results
write_csv(results,'./test.csv');



















