# Person and PPE Deetction

This codebase is for Person Detection and followed by PPE Detection 

## Generating required dataset for both the tasks

1. The provided labels are in Pascal VOC format. So, "pascal_to_yolo.py" script can be used to convert the labels to yolo format.
2. Generate labels for person detection using below command -
 ```
python3 pascal_to_yolo.py --input_images datasets/images/ --input_labels datasets/labels/ --output_path datasets/person_dataset/labels/
```
3. Copy the "datasets/images" folder to datasets/person_dataset/. So the dataset structure for person detection is
```
datasets
  person_dataset
      images
      labels
```
4.  For PPE Detection, we want to train the model on cropped images. So, we need to generate labels for such cropped images as well. For this purpose, I first convert ppe labels from voc format to yolo and save it in labels_ppe_all, use the command as below
```
   python3 pascal_to_yolo.py --input_images datasets/images/ --input_labels datasets/labels/ --output_path datasets/labels_ppe_all/
```
Be sure to comment line 17 in "pascal_to_yolo.py" and uncomment line 19 before running the above command.

5. To generate cropped images and corresponding labels for PPE detection, run below command

```
python3 generate_data_ppe.py --input_images datasets/person_dataset/images/ --input_person_labels datasets/person_dataset/labels/ --input_ppe_labels datasets/labels_ppe_all/ --cropped_images datasets/ppe_dataset/images/ --output_ppe_labels datasets/ppe_dataset/labels/ --ground_truth_vis datasets/ppe_dataset/ground_truth_vis/
```
This will generate images, labels and also visualization of the labels on images in datasets/ppe_dataset folder. 

6. The dataset structure for PPE detection will be as follows
```
datasets
  ppe_dataset
      images
      labels
      ground_truth_vis
```

## Training 

1. For training person detection model, run below command

```
python3 train.py --yaml person_detection_data.yaml --model yolov8m.pt --image_size 640 --epochs 100 --batch_size 8 --name yolov8m_person_100epochs
```

The training plots and results can be seen at runs/detect/yolov8m_person_100epochs

![results](https://github.com/manikamakam/ppe_detection/assets/48440422/3ccaa06f-cb9c-4703-a054-c9ea4e4802f4)
![confusion_matrix](https://github.com/manikamakam/ppe_detection/assets/48440422/c02eaac3-13b7-4983-aec9-c37e3344a85a)

#### Achieved mAP50 of 0.99384 and mAP50-95 of 0.94647


2. For training, PPE detection model, run below command

```
python3 train.py --yaml ppe_detection_data.yaml --model yolov8m.pt --image_size 640 --epochs 200 --batch_size 8 --name yolov8m_ppe_200epochs
```
The training plots and results can be seen at runs/detect/yolov8m_ppe_200epochs
![results](https://github.com/manikamakam/ppe_detection/assets/48440422/ef7e5a9a-4237-480f-bfdb-65567b84b2b2)

![confusion_matrix](https://github.com/manikamakam/ppe_detection/assets/48440422/77e12937-0dca-4e08-86e1-0d566c696ffd)

#### Achieved mAP50 of 0.9937 and mAP50-95 of 0.9704

## Inference 

Inference script will take input images, run prediction from person detection model to generate cropped images and then runs ppe deetction model on the cropped images. All results are stored in "output/". Run below command - 
```
 python3 inference.py --input_dir datasets/images/ --output_dir output/ --person_model weights/best_person.pt --ppe_model weights/best_ppe.pt
```
![0](https://github.com/manikamakam/ppe_detection/assets/48440422/3c4bab90-c53d-4d48-872f-6eea07ea41ea)
![1](https://github.com/manikamakam/ppe_detection/assets/48440422/d62b52be-0591-4548-a1a6-2fa36863c78f)
![2](https://github.com/manikamakam/ppe_detection/assets/48440422/6cb16b97-a9c4-420a-896d-c2ba0a7cac43)


## Algorithm and observations
#### Person Deetction 

  1. For Person detection, it is possible that the data is very less - that is trained using only 416 images. So, it is possible that the model has been over-fitted.
  2. For some images, unwanted portion has been detected as person. So there are some false positives
  3. I have tried training with multiple models - yolo nano, yolo small and yolo medium model. I have observed best results were observed with medium model. The other training runs are stored at runs/deetct/archive for reference
  4. The best model (weights/best_person.pt) achieved mAP50 of 0.99384 and mAP50-95 of 0.94647

#### PPE Detection 
  1. First challenge was to generate labels of the cropped images. I used Albumentations library to generate labels of transformed/cropped images.
  2. I have tried training with multiple models - yolo nano, yolo small and yolo medium model. I have observed best results were observed with medium model.
  3. I have also tried to ignore the 'ear-protector' and 'safety-harness' class as instances of it are not present in the dataset. However, the results don't change really as yolo calculates loss for the classes observed in the dataset.
  4. The total cropped images were 1284. However 3 of them were very small, so it was ignored during training. Thus model was trained with 1281 images.
  5. Some of the cropped images were veryt blurry, so this might have been an issue for the model in some cases. 
  6. The best model (weights/best_ppe.pt) achieved mAP50 of 0.9937 and mAP50-95 of 0.9704
