import json
import shutil
import os
import cv2
import numpy as np
from pathlib import Path
from random import shuffle


def save_cmu_split_indices(train_per=0.95, save_dir=(Root_Dir / "src/HandEst/data/Public_HandData/CMU_Panoptic/split_indices")):
    cmu_labels_path = (Root_Dir / "src/HandEst/data/Public_HandData/CMU_Panoptic/hand143_panopticdb/hands_v143_14817.json").as_posix()
    labels = json.load(open(cmu_labels_path,"r"))
    labels_list = labels['root']
    num_indices = len(labels_list)
    indices = np.arange(start=0, stop=num_indices)
    shuffle(indices)

    train_indices = indices[0:int(num_indices*train_per)]
    val_indices = indices[int(num_indices*train_per)::]

    train_per_int = int(train_per*100)
    train_sp = Path(save_dir) / f"train_indices_{train_per_int}.npy"
    np.save(train_sp.as_posix(), train_indices)
    val_sp = Path(save_dir) / f"val_indices_{100-train_per_int}.npy"
    np.save(val_sp.as_posix(),val_indices)

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
def convert_cmu_data_2_yolo(train_indices, val_indices):
    #right hand class is custom yolo class
    hand_class = 0


    #load cmu labels
    cmu_labels_path = (Root_Dir / "src/HandEst/data/Public_HandData/CMU_Panoptic/hand143_panopticdb/hands_v143_14817.json").as_posix()
    labels = json.load(open(cmu_labels_path,"r"))
    cmu_image_dir = (Root_Dir / "src/HandEst/data/Public_HandData/CMU_Panoptic/hand143_panopticdb/imgs").as_posix()

    #create yolo dataset dir
    yolo_dir = Path("/home/inseer/Research/yolov7/CMU_Panoptic")
    yolo_dir.mkdir(exist_ok=True ,parents=True)
    yolo_img_dir = yolo_dir / "images"
    yolo_img_dir.mkdir(exist_ok=True)
    (yolo_img_dir / "train").mkdir(exist_ok=True)
    (yolo_img_dir / "val").mkdir(exist_ok=True)
    yolo_label_dir = yolo_dir / "labels"
    yolo_label_dir.mkdir(exist_ok=True)
    (yolo_label_dir / "train").mkdir(exist_ok=True)
    (yolo_label_dir / "val").mkdir(exist_ok=True)


    labels_list = labels['root']

    #create train/val catalogues
    train_cat_path = (yolo_dir / "train.txt").as_posix()
    val_cat_path  = (yolo_dir / "val.txt").as_posix()
    train_cat = open(train_cat_path,"w")
    val_cat = open(val_cat_path,"w")


    for e_i, entry in enumerate(labels_list):
        #get cmu annotation data
        img_path = Path(entry['img_paths'])
        img_ext = img_path.suffix
        img_name = img_path.with_suffix('').name
        img_h, img_w = entry['img_height'], entry['img_width']
        bbox_center = entry['objpos']
        bbox_scale = entry['scale_provided']
        #new data for yolo txt file
        x_center, y_center = np.round(np.asarray(bbox_center)).astype(dtype=int)
        square_len = np.round(175*bbox_scale).astype(dtype=int)
        box_h , box_w = 2*square_len, 2*square_len
        #normalize data
        x_center = np.round(x_center / img_w, 4)
        y_center = np.round(y_center / img_h, 4)
        box_w = np.round(box_w / img_w, 4)
        box_h = np.round(box_h / img_h, 4)
        #write data to txt file
        val_ann = True if e_i in val_indices else False
        if val_ann:
            txt_path = (yolo_label_dir / f"val/{img_name}").with_suffix('.txt')
        else:
            txt_path = (yolo_label_dir / f"train/{img_name}").with_suffix('.txt')

        with open(txt_path.as_posix(), 'w') as file:
            file.write(f"{hand_class} {x_center} {y_center} {box_w} {box_h}\n")

        #copy image
        img_path = (Path(cmu_image_dir) / img_name).with_suffix(img_ext)
        if val_ann:
            new_img_path = (yolo_img_dir / f"val/{img_name}").with_suffix(img_ext)
            rel_path = f"./images/val/{img_name}{img_ext}"
            val_cat.write(f"{rel_path}\n")

        else:
            new_img_path = (yolo_img_dir / f"train/{img_name}").with_suffix(img_ext)
            rel_path = f"./images/train/{img_name}{img_ext}"
            train_cat.write(f"{rel_path}\n")

        shutil.copy(src=img_path.as_posix(), dst=new_img_path.as_posix())

        #test annotation
        # image = cv2.imread(img_path.as_posix())
        # visualize_box(image, x_center,y_center,box_w,box_h)
        # print(entry["img_paths"])
    train_cat.close()
    val_cat.close()




if __name__ == "__main__":
    train_indices = np.load(
        (Root_Dir / "src/HandEst/data/Public_HandData/CMU_Panoptic/split_indices/train_indices_95.npy").as_posix(),
        allow_pickle=True)
    val_indices = np.load(
        (Root_Dir / "src/HandEst/data/Public_HandData/CMU_Panoptic/split_indices/val_indices_5.npy").as_posix(),
        allow_pickle=True)

    convert_cmu_data_2_yolo(train_indices,val_indices)