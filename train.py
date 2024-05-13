from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser(
                    prog='Training script',
                    description='Train person detection model or ppe detection model',
                    epilog='train script')

parser.add_argument('--yaml',type = str, default= "ppe_detection_data.yaml")   #use 'person_detection_data.yaml' for person detection
parser.add_argument('--model',type = str, default= "yolov8s.pt")                  # choose 'yolov8n.pt' or 'yolov8s.pt' or 'yolov8m.pt'
parser.add_argument('--image_size',type = int, default= 640)      
parser.add_argument('--epochs',type = int, default= 100)      
parser.add_argument('--batch_size',type = int, default= 1)     
parser.add_argument('--name',type = str, default= 'yolov8s_ppe')               # choose name coresponding to model chosen; results will be stored in this folder name

args = parser.parse_args()

# Load the model.
# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8s.pt')
# model = YOLO('yolov8m.pt')
model = YOLO(args.model)
 
# Training
results = model.train(
   data=args.yaml,
   imgsz=args.image_size,
   epochs=args.epochs,
   batch=args.batch_size,
   name=args.name
)

# results = model.train(
#    data='person_detection_data.yaml',
#    imgsz=640,
#    epochs=30,
#    batch=8,
#    name='yolov8n_person'
# )

# results = model.train(
#    data='ppe_detection_data.yaml',
#    imgsz=640,
#    epochs=100,
#    batch=1,
#    name='yolov8s_ppe'
# )