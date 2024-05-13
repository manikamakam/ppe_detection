import glob
import os
import xml.etree.ElementTree as ET
import argparse
import cv2
import albumentations as A
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
                    prog='Pascal VOC format to YOLO format',
                    description='Run script to convert annotations from pascal format to yolo format',
                    epilog='VOC to YOLO')

parser.add_argument('--input_images',default= "datasets/person_dataset/images/")   
parser.add_argument('--input_person_labels',default= "datasets/person_dataset/labels/")   
parser.add_argument('--input_ppe_labels',default= "datasets/labels_ppe_all/")   
parser.add_argument('--cropped_images',default= "datasets/ppe_dataset/cropped_images/")          
parser.add_argument('--output_ppe_labels',default= "datasets/ppe_dataset/output_labels_ppe/")    
parser.add_argument('--ground_truth_vis',default= "datasets/ppe_dataset/ground_truth_vis/")   

args = parser.parse_args()

category_id_to_name = {0: 'hard-hat', 1: 'gloves', 2: 'mask',3: 'glasses', 4: 'boots', 5: 'vest', 6: 'ppe-suit', 7: 'ear-protector', 8:'safety-harness'}
# per_classes = ['person']

BOX_COLOR = (255, 0, 0) # Red
TEXT_COLOR = (255, 255, 255) # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)

    # print(x_min, y_min, x_max, y_max)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)
    return img

def getImagesInDir(dir_path):
    image_list = []
    for filename in glob.glob(dir_path + '/*.jpg'):
        image_list.append(filename)

    return image_list

def convert_to_yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

image_paths = getImagesInDir(args.input_images)

if not os.path.exists(args.cropped_images):
        os.makedirs(args.cropped_images)
if not os.path.exists(args.output_ppe_labels):
        os.makedirs(args.output_ppe_labels)
if not os.path.exists(args.ground_truth_vis):
        os.makedirs(args.ground_truth_vis)
# image_paths = ["datasets/person_dataset/images/-1477-_png_jpg.rf.bac8d06edca64da17ced23797d0e2339.jpg"]
for image_path in image_paths:
    basename = os.path.basename(image_path)
    basename_no_ext = os.path.splitext(basename)[0]

    person_ann_file = open(args.input_person_labels +  basename_no_ext + '.txt', 'r')
    persons = person_ann_file.readlines()

    ppe_ann_file = open(args.input_ppe_labels +  basename_no_ext + '.txt', 'r')
    ppe_anns = ppe_ann_file.readlines()

    img = cv2.imread(image_path)
    img_h, img_w, _ = img.shape
    count = 0

    boxes = []
    # category_ids = []
    for ppe in ppe_anns:
        b = ppe.split()
        cls, x_cen, y_cen, w, h = b
        # print(img_w* ((float(w)/2) + float(x_cen)))
        # print(int(img_w* ((float(w)/2) + float(x_cen))))
        x_max = int(img_w* ((float(w)/2) + float(x_cen)) + 1)
        x_min = int(x_max - (float(w)* img_w) + 1)
        y_max = int(img_h* ((float(h)/2) + float(y_cen)) + 1)
        y_min = int(y_max - (float(h)* img_h) + 1)
        width = x_max - x_min 
        height = y_max - y_min 
        boxes.append([cls, x_min, y_min, width, height])
        # boxes.append([cls, float(x_cen), float(y_cen), float(w), float(h)])
        # category_ids.append(int(b[0]))
    print("************************************************************************")
    print(image_path)
    # print(len(boxes))
    # print(len(category_ids))
    print("persons: ", len(persons))

    for per in persons:
        print("per: ", per)
        b = per.split()
        # print(b)
        _, x_cen, y_cen, w, h = b
        x_cen = float(x_cen)
        y_cen = float(y_cen)
        w = float(w)
        h = float(h)
    
        x_max = int(img_w* ((float(w)/2) + float(x_cen))+ 1)
        x_min = int(x_max - (float(w)* img_w) + 1)
        y_max = int(img_h* ((float(h)/2) + float(y_cen)) + 1)
        y_min = int(y_max - (float(h)* img_h) + 1)

        # print(x_min, y_min, x_max, y_max)
        if x_max-x_min <= img_w and y_max-y_min <= img_h and 0 <= x_min <= img_w and 0 <= x_max <= img_w and 0 <= y_min <= img_h and 0 <= y_max <= img_h:
            transform = A.Compose(
                            [A.Crop(x_min=x_min, y_min = y_min, x_max= x_max, y_max=y_max, always_apply=False, p=1.0)],
                            bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
                        )
            # print("xxxxx", transform)
            filtered_boxes = []
            category_ids = []
            # print(boxes)
            for box in boxes:
                if box[1] >= x_min-1 and box[2] >= y_min-1 and box[3] <= x_max - x_min and box[4] <= y_max -y_min:
                # if 0 < box[1] <= 1 and 0 < box[2] <= 1 and 0 < box[3] <= 1 and 0 < box[4] <= 1:
                    filtered_boxes.append([box[1], box[2], box[3], box[4]])
                    category_ids.append(int(box[0]))
            # print("filtered boxes", filtered_boxes)

            transformed = transform(image=img, bboxes=filtered_boxes, category_ids=category_ids)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_labels = transformed['category_ids']
            print("transformed_bboxes: ", transformed_bboxes)
            print("transformed_class_labels: ", transformed_class_labels)

            result_img = visualize(
                transformed['image'],
                transformed['bboxes'],
                transformed['category_ids'],
                category_id_to_name,
            )
            
            cv2.imwrite(args.cropped_images + basename_no_ext + "_" + str(count) + ".jpg", transformed['image'])
            cv2.imwrite(args.ground_truth_vis + basename_no_ext + "_" + str(count) + ".jpg", result_img)            
            
            trans_h, trans_w, _ = transformed['image'].shape
            out_file = open(args.output_ppe_labels + basename_no_ext + "_" + str(count)  + '.txt', 'w')
            for i in range(len(transformed['bboxes'])):
                t_box = transformed['bboxes'][i]
                bb = convert_to_yolo((trans_w,trans_h), [t_box[0], t_box[0] + t_box[2], t_box[1], t_box[1] + t_box[3]])
                out_file.write(str(transformed['category_ids'][i]) + " " + " ".join([str(a) for a in bb]) + '\n')
            count = count + 1
            print("`````````````````````````````````````````````````````````````````````````")
                 
                 
    