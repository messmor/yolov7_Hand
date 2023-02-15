import json
import shutil
import os
import cv2
import numpy as np
import imagesize
from pathlib import Path
from random import shuffle
from copy import deepcopy


def fill_nums(number):
    """enter a number with less than 12 digits, outputs"""
    """a number with 12 digits, padded with zeros."""
    number = str(number)
    assert len(number) <= 12
    pad = 12 - len(number)
    new_number = ''.join(["0" for i in range(pad)]) + number

    return new_number
def visualize_box(image, x_center, y_center, box_w, box_h):
    img_h,img_w, _ = image.shape
    x_center *= img_w
    y_center *= img_h
    box_h *= img_h
    box_w *= img_w
    pt1 = (int(x_center-(box_w/2)),int(y_center-(box_h/2)))
    pt2 = (int(x_center+(box_w/2)),int(y_center+(box_h/2)))

    image = cv2.rectangle(image, pt1=pt1,pt2=pt2, color=[0,255,0], thickness=3)
    cv2.imshow('test', image)
    cv2.waitKey()
    return

def convert_2_yolo(bbox, img_w, img_h):
    """Converts a bbox in the format [x_min,x_max,y_min,y_max] to a normalized"""
    """yolo format of [x_center, y_center, box_h, box_w] """

    x_min, x_max, y_min, y_max = bbox
    # convert box to yolo format
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    box_h = abs(y_max - y_min)
    box_w = abs(x_max - x_min)

    # normalize data
    x_center = np.round(x_center / img_w, 4)
    y_center = np.round(y_center / img_h, 4)
    box_w = np.round(box_w / img_w, 4)
    box_h = np.round(box_h / img_h, 4)

    return [x_center, y_center, box_w, box_h]
def convert_COCO_Hand_data_2_yolo(val_indices, Root_COCO_Hand, yolo_root, visualize=False):
    print(f"starting data conversion for {Root_COCO_Hand}! ")
    #right hand class is custom yolo class
    hand_class = 0
    #load cmu labels
    Root_COCO_Hand = Path(Root_COCO_Hand)
    COCO_labels_path = (Root_COCO_Hand / "COCO-Hand-Big/COCO-Hand-Big_annotations.txt").as_posix()
    COCO_image_dir = (Root_COCO_Hand / "COCO-Hand-Big/COCO-Hand-Big_Images")

    #create yolo dataset dir
    yolo_dir = Path(yolo_root)
    if not yolo_dir.is_dir():
        raise NameError(f"yolo_root {yolo_root} is not a directory")
    yolo_dir = yolo_dir / "COCO-Hand"
    yolo_dir.mkdir(exist_ok=True, parents=True)
    yolo_img_dir = yolo_dir / "images"
    yolo_img_dir.mkdir(exist_ok=True)
    (yolo_img_dir / "train").mkdir(exist_ok=True)
    (yolo_img_dir / "val").mkdir(exist_ok=True)
    yolo_label_dir = yolo_dir / "labels"
    yolo_label_dir.mkdir(exist_ok=True)
    (yolo_label_dir / "train").mkdir(exist_ok=True)
    (yolo_label_dir / "val").mkdir(exist_ok=True)

    ann_txt = open(COCO_labels_path,'r')
    labels_list = [line.split('\n')[0] for line in ann_txt if line != '\n']

    #create train/val catalogues
    train_cat_path = (yolo_dir / "train.txt").as_posix()
    val_cat_path  = (yolo_dir / "val.txt").as_posix()
    train_cat = open(train_cat_path,"a")
    val_cat = open(val_cat_path,"a")


    for e_i, entry in enumerate(labels_list):
        data = entry.split(',')
        if min([int(x) for x in data[1:5]]) < 0:
            continue
        ###note data has the form [image_name, xmin, xmax, ymin, ymax, x1, y1, x2, y2, x3, y3, x4, y4]
        ###get cmu annotation data
        img_path = COCO_image_dir / data[0]
        img_ext = img_path.suffix
        img_name = img_path.with_suffix('').name
        img_w, img_h = imagesize.get(img_path.as_posix())
        x_min,x_max,y_min,y_max = [int(data[i]) for i in range(1,5)]
        x_center, y_center, box_w, box_h = convert_2_yolo([x_min, x_max,y_min,y_max],img_w,img_h)

        #write data to txt file
        val_ann = True if e_i in val_indices else False
        if val_ann:
            txt_path = (yolo_label_dir / f"val/{img_name}").with_suffix('.txt')
        else:
            txt_path = (yolo_label_dir / f"train/{img_name}").with_suffix('.txt')

        with open(txt_path.as_posix(), 'a') as file:
            file.write(f"{hand_class} {x_center} {y_center} {box_w} {box_h}\n")

        #copy image
        if val_ann:
            new_img_path = (yolo_img_dir / f"val/{img_name}").with_suffix(img_ext)
            rel_path = f"./images/val/{img_name}{img_ext}"
            val_cat.write(f"{rel_path}\n")

        else:
            new_img_path = (yolo_img_dir / f"train/{img_name}").with_suffix(img_ext)
            rel_path = f"./images/train/{img_name}{img_ext}"
            train_cat.write(f"{rel_path}\n")

        shutil.copy(src=img_path.as_posix(), dst=new_img_path.as_posix())

        ###test annotation
        if visualize:
            image = cv2.imread(img_path.as_posix())
            visualize_box(image, x_center,y_center,box_w,box_h)

    train_cat.close()
    val_cat.close()







