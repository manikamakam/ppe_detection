from ultralytics import YOLO
import cv2
# import torch
# from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import os
import argparse

parser = argparse.ArgumentParser(
                    prog='Inference',
                    description='Run inference from two models and save results',
                    epilog='PPE Detection')

parser.add_argument('--input_dir',default= "datasets/person_dataset/images/")           
parser.add_argument('--output_dir',default= "output/")      
parser.add_argument('--person_model',default= "weights/best_person.pt")           
parser.add_argument('--ppe_model',default= "weights/best_ppe.pt") 

args = parser.parse_args()

model = YOLO(args.person_model)
img_list = list(filter(lambda x: '.jpg' in x, os.listdir(args.input_dir)))
# print(len(img_list))
cnt = 0
print("Starting prediction from person detection model")
for path in img_list:
    print(cnt)
    print("input image path: ", args.input_dir + path)
    img = cv2.imread(args.input_dir + path)
    results = model.predict(img, verbose = False)
    boxes = results[0].boxes
    count = 0
    out_path = args.output_dir + "person/" + path + "/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for box in boxes:
        b = box.xyxy[0].cpu() # get box coordinates in (left, top, right, bottom) format
        b = b.numpy()
        c = box.cls
        cropped_image = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]

        print("output image path: ", out_path + str(count) + ".jpg")
        cv2.imwrite(out_path + str(count) + ".jpg", cropped_image)
        print("`````````````````````````````````````````````````````````````")
        count = count + 1
    cnt = cnt + 1

print("Starting prediction from ppe detection model")
person_path = args.output_dir + "person/"
ppe_path = args.output_dir + "ppe/"
ppe_model = YOLO(args.ppe_model)
for path in img_list:
    images = list(filter(lambda x: '.jpg' in x, os.listdir(person_path + path)))
    if not os.path.exists(ppe_path + path + "/"):
        os.makedirs(ppe_path + path + "/")
    for img in images:
        print("input image path: ", person_path + path + "/" + img)
        image = cv2.imread(person_path + path + "/" + img)
        results = ppe_model.predict(image, verbose = False)
        boxes = results[0].boxes
        for box in boxes:
            # print("boxessssssssssssssssss")
            b = box.xyxy[0].cpu() # get box coordinates in (left, top, right, bottom) format
            
            b = b.numpy()
            # print(b[0])
            c = box.cls.cpu().numpy()

            if (int(c[0])== 0):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (255, 0, 0) , 1) 
                image = cv2.putText(image, 'hard-hat', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (255, 0, 0), 0, cv2.LINE_AA) 
            if (int(c[0])== 1):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (0, 255, 0) , 1) 
                image = cv2.putText(image, 'gloves', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 255, 0), 0, cv2.LINE_AA) 
            if (int(c[0])== 2):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (0, 0, 255) , 1) 
                image = cv2.putText(image, 'mask', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 0, 255), 0, cv2.LINE_AA) 
            if (int(c[0])== 3):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (255, 255, 0) , 1) 
                image = cv2.putText(image, 'glasses', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (255, 255, 0), 0, cv2.LINE_AA) 
            if (int(c[0])== 4):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (255, 0, 255) , 1) 
                image = cv2.putText(image, 'boots', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (255, 0, 255), 0, cv2.LINE_AA) 
            if (int(c[0])== 5):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (0, 255,255) , 1) 
                image = cv2.putText(image, 'vest', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 255,255), 0, cv2.LINE_AA) 
            if (int(c[0])== 6):
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (255, 255, 255) , 1) 
                image = cv2.putText(image, 'ppe-suit', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (255, 255, 255), 0, cv2.LINE_AA) 
            # if (int(c[0])== 7):
            #     image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (0, 0, 0) , 1) 
            #     image = cv2.putText(image, 'ear-protector', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (0, 0, 0), 0, cv2.LINE_AA) 
            # if (int(c[0])== 8):
            #     image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]),int(b[3])), (150, 150, 150) , 1) 
            #     image = cv2.putText(image, 'safety-harness', (int(b[0]),int(b[3]+10)) , cv2.FONT_HERSHEY_SIMPLEX,  0.2, (150, 150, 150), 0, cv2.LINE_AA) 
            
        cv2.imwrite(ppe_path + path + "/" + img, image)
        print("output image path: ", ppe_path + path + "/" + img)
        print("`````````````````````````````````````````````````````````````")