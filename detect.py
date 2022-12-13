
import av
import gzip
import time

import numpy as np
import torch
from math import ceil
from numpy import random
from pathlib import Path
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

class Output_Video(object):
    '''writes video output '''

    def __init__(self, output_path, frame_size, fps, codec='h264'):
        self.codec = av.Codec(codec, 'w')
        self.frame_size = frame_size
        self.fps = fps
        self.output = av.open(output_path, 'w')
        self.stream = self.output.add_stream(self.codec, ceil(self.fps))
        self.stream.width = frame_size[1]
        self.stream.height = frame_size[0]
        self.stream.options = {'crf': '30'}

    def write_frame(self, frame):
        frame = av.VideoFrame.from_ndarray(frame, format='bgr24')
        packet = self.stream.encode(frame)
        self.output.mux(packet)
    def close_stream(self):
        self.output.close()

def detect(source, weights, imgsz, device, conf_thres=0.25, iou_thres=0.45, classes=[1], write_video=False, save_pred=False):
    #ensure source video path is valid
    if not Path(source).is_file():
        FileNotFoundError(f"source video path {source} does not exist!")

    # Directories
    save_dir = Path(source).parent
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)

    #get image size p

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    video_frame_size = dataset.video_frame_size
    print("video framesize:", video_frame_size)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # create video output
    if write_video:
        video_path = Path(save_dir) / "yolo_video.mp4"
        print(f"video output saved here: {video_path.as_posix()}")
        fps = 30 if dataset.fps is None else dataset.fps
        vid_writer = Output_Video(output_path=video_path.as_posix(), frame_size=video_frame_size, fps=fps)
    #save the predictions as a list
    all_predictions = []
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=None)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                image_predictions = []
                for *xyxy, conf, cls in reversed(det):
                    # Save results
                    xyxy = [int(c.cpu()) for c in xyxy]
                    image_predictions.append(xyxy+[round(float(conf.detach().cpu()),4), int(cls.detach().cpu())])
                    if write_video:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            all_predictions.append(image_predictions)



        if write_video:
            vid_writer.write_frame(im0)


    if write_video:
        vid_writer.close_stream()

    #save predictions
    print("test all predictions:", all_predictions[0:10])
    predictions_path = (save_dir / "yolov7_predictions.npy.gz").as_posix()
    file = gzip.GzipFile(predictions_path, "w")
    np.save(file, all_predictions)


    print(f'Done. ({time.time() - t0:.3f}s)')

    return all_predictions


if __name__ == '__main__':

    weights = "/home/inseer/engineering/yolov7_Hand/weights/Hands_COCO_Dataset/best.pt"
    source = "/home/inseer/data/App_Main_Testing/Hand_Videos/test_vid8/vid.MOV"
    imgsz = 384
    device = '0'
    write_video = False

    with torch.no_grad():
            detect(source, weights,imgsz,device,write_video)
