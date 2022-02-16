import csv
import cv2
import numpy as np
import torch
import math
import re
import json
from torch.utils.data import Dataset
import yaml
import random

with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

re_size = config["re_size"]
scale_factor = config["scale_factor"]
sampling_frac_train = tuple(config["sampling_frac_train"])
sampling_frac_valid = tuple(config["sampling_frac_valid"])


class MotionDetDataset(Dataset):

    def __init__(self, labels_file, datatype):

        print(f"Initializing {datatype} dataset...")

        self.labels = list(csv.DictReader(labels_file))
        allnpy = {}
        # Load all images.
        len_label = len(self.labels)
        for i, label in enumerate(self.labels, start=1):
            if i % 1000 == 0:
                print(f"\tLoaded {datatype} .npy files {i}/{len_label}")
            npy_path = label["npypath"]
            if npy_path not in allnpy:
                allnpy[npy_path] = np.load(npy_path)
        print(f"Finished loading {datatype} .npy files.")

        self.boxlst = []
        self.sampling_count = 0
        self.datatype = datatype
        self.sampling_frac = sampling_frac_train if self.datatype == "train" else sampling_frac_valid
        
        # Deal with obj one by one.
        for label in self.labels:
            
            # Downsampling for still obj.
            # !!!Need to be commented out during visualization!!!
            if datatype != "infer" and label["motionFlag"] == '0':
                self.sampling_count += 1
                if self.sampling_count > self.sampling_frac[0] and self.sampling_count < self.sampling_frac[1]:
                    continue
                elif self.sampling_count == self.sampling_frac[1]:
                    self.sampling_count = 0
                    continue
            
            img_data = allnpy[label["npypath"]]
            left = float(label["xmin"])
            top = float(label["ymin"])
            right = float(label["xmax"])
            bottom = float(label["ymax"])
            # Reshape to square.
            diff = ((right - left) - (bottom - top)) / 2
            if diff > 0:
                top -= diff
                bottom += diff
            else:
                diff = abs(diff)
                left -= diff
                right += diff
            # Scale up to {scale_factor} times.
            scale = (right - left) * scale_factor / 2
            top -= scale
            bottom += scale
            left -= scale
            right += scale
            # Padding for edge cases.
            height, width, _ = img_data.shape
            if top < 0 or bottom > height or left < 0 or right > width:
                pad_size = abs(math.floor(
                    min(top, height - bottom, left, width - right)))
                img_pad = np.pad(
                    img_data, ((pad_size,), (pad_size,), (0,)), "edge")
            else:
                pad_size = 0
                img_pad = img_data
            top = math.floor(top+pad_size)
            bottom = math.ceil(bottom+pad_size)
            left = math.floor(left+pad_size)
            right = math.ceil(right+pad_size)
            box = img_pad[top:bottom+1, left:right+1, :]

            box = cv2.resize(box, (re_size, re_size),
                             interpolation=cv2.INTER_LINEAR)
            gt = np.array(int(label["motionFlag"]))
            self.boxlst.append(tuple((box, gt)))

        print(f"Finished processing {datatype} flow files.")
        print(f"Finished initializing {datatype} dataset.")

    def __len__(self):
        return len(self.boxlst)

    def __getitem__(self, idx):
        box, gt = self.boxlst[idx]

        # Random flip with -1 * u, i.e. horizontal velocity.
        if random.uniform(0, 1) > 0.5:
            box = np.flip(box, axis=1)
            box[..., 0] = box[..., 0] * -1
        
        box = np.transpose(box, (2, 0, 1))  
        box_tensor = torch.from_numpy(box.copy()).type(torch.float32)
        
        return box_tensor, torch.from_numpy(gt)
    
