import cv2
import json
import numpy as np
import imagesize
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm


def fill_nums(number):
    """enter a number with less than 12 digits, outputs"""
    """a number with 12 digits, padded with zeros."""
    number = str(number)
    assert len(number) <= 12
    pad = 12 - len(number)
    new_number = ''.join(["0" for i in range(pad)]) + number

    return new_number

def visualize_bbox(image, bbox, format="tlwh"):
    if format == "tlwh":
        left, top, width, height = bbox
        pt1 = (int(left), int(top))
        pt2 = (int(left+width), int(top+height))

    elif format == "yolo":
        x_center, y_center, box_w, box_h = bbox
        pt1 = (int(x_center-box_w/2),int(y_center-box_h/2))
        pt2 = (int(x_center+box_w/2),int(y_center+box_h/2))
    else:
        ValueError(f'bbox format must be either tlwh or yolo. Not {format}!')
    image = cv2.rectangle(image, pt1=pt1, pt2=pt2, color=[0, 0, 255], thickness=3)


    return image

def convert_2_yolo(bbox, img_w, img_h, normalize=True):
    """Converts a bbox in the format [x_min,x_max,y_min,y_max] to a normalized"""
    """yolo format of [x_center, y_center, box_h, box_w] """

    left, top, width, height = bbox
    # convert box to yolo format
    x_center = left +(width / 2)
    y_center = top + (height / 2)
    box_h = height
    box_w = width

    # normalize data
    if normalize:
        x_center = np.round(x_center / img_w, 4)
        y_center = np.round(y_center / img_h, 4)
        box_w = np.round(box_w / img_w, 4)
        box_h = np.round(box_h / img_h, 4)

    return [x_center, y_center, box_w, box_h]
def convert_COCO_data_2_yolo(COCO_root,split="train"):
    """TODO: create a method that will go through the COCO dataset list of annotations. Anytime the"""
    """there is a person id, it checks to see if there is a hand annotation in COCO_Hand_Dataset."""
    """if there is then an annotation txt for the given image in the COCO_Hand_Dataset (a copy lives """
    """COCO-Hand-Person) and the person annotation is written to the text. This will create a list of labels"""
    """that can be used to train a yolo network that detects hand and people at the same time. Reducing the"""
    """run time of the total ml_pipeline."""

    assert split == "train" or split =="val"
    Data_Root = Path(COCO_root)
    if not Data_Root.is_dir():
        IsADirectoryError(f"{Data_Root.as_posix()} is not a directory!")

    data = json.load(open((Data_Root / f"coco_wholebody_{split}_v1.0.json").as_posix() ,"r"))
    label_dir = Data_Root / f"labels/{split}"
    Path(label_dir).mkdir(exist_ok=True,parents=True)

    for entry in tqdm(data["annotations"]):
        image_id = deepcopy(entry['image_id'])
        image_name = f"{fill_nums(image_id)}.jpg"
        image_rel_path = f"{split}2017/{image_name}"
        image_path = Data_Root / image_rel_path
        label_txt_path = label_dir / f"{Path(image_name).with_suffix('.txt').as_posix()}"
        if not Path(image_path).is_file():
            print(f"skipping image {image_name}, not found!")
            continue

        img_w, img_h = imagesize.get(image_path.as_posix())
        person_box = convert_2_yolo(deepcopy(entry['bbox']), img_w, img_h)
        face_box = convert_2_yolo(deepcopy(entry['face_box']), img_w, img_h)
        lhand_box = convert_2_yolo(deepcopy(entry['lefthand_box']), img_w, img_h)
        rhand_box = convert_2_yolo(deepcopy(entry['righthand_box']), img_w, img_h)

        boxes = [(person_box,0), (lhand_box,1),(rhand_box,1),(face_box,2)]

        for box, class_id in boxes:
            if np.amax(box) <=0:
                continue
            x_center, y_center, box_w, box_h = box
            with open(label_txt_path.as_posix(), 'a') as file:
                file.write(f"{class_id} {x_center} {y_center} {box_w} {box_h}\n")


    return



if __name__ == "__main__":
    COCO_root = "/media/mitchell/ssd2/COCO/"
    convert_COCO_data_2_yolo(COCO_root, split="val", catalog_only=True)
    convert_COCO_data_2_yolo(COCO_root, split="train", catalog_only=True)