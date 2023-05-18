import os
import json
import time

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box
from tqdm import tqdm
from PIL import Image

dir_origin_path = "my_yolo_dataset/test/images/"
dir_save_path = "img_out/"



img_names = os.listdir(dir_origin_path)

def main():
    img_size = 416
    cfg = "cfg/MCDet.cfg"
    weights = "weights/20230509-113944.txtbest.pt"
    json_path = "./data/pascal_voc_classes.json"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)

    json_file = open(json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)["model"])
    model.to(device)

    model.eval()

    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)

            with torch.no_grad():
                # init

                img_o = cv2.imread(image_path)  # BGR
                assert img_o is not None, "Image Not Found "

                img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device).float()
                img /= 255.0  # scale (0, 255) to (0, 1)
                img = img.unsqueeze(0)  # add batch dimension

                #t1 = torch_utils.time_synchronized()
                pred = model(img)[0]  # only get inference result
                #t2 = torch_utils.time_synchronized()
                #print(t2 - t1)

                pred = utils.non_max_suppression(pred, conf_thres=0.35, iou_thres=0.5, multi_label=True)[0]

                if pred is not None:
                    # process detections
                    pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
                    print(pred.shape)

                    bboxes = pred[:, :4].detach().cpu().numpy()
                    scores = pred[:, 4].detach().cpu().numpy()
                    classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1

                    img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)
                    plt.imshow(img_o)
                    #plt.show()
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    img_o.save(os.path.join(dir_save_path, img_name))
                if pred is None:
                    cv2.imwrite(os.path.join(dir_save_path, img_name), img_o)


if __name__ == "__main__":
    main()
