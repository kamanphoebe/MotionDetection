import sys
sys.path.append('core')

import argparse
import cv2
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import os
from glob import glob

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, flowpath, rfpath):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=1)

    flo = cv2.cvtColor(flo, cv2.COLOR_BGR2RGB)
    cv2.imwrite(flowpath, flo)
    cv2.imwrite(rfpath, img_flo[:, :, [2,1,0]])
    

def demo(args):

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    raw_path = os.path.join(args.repo_path, "custom_demo", "custom_raw")
    flow_path = os.path.join(args.repo_path, "custom_demo", "custom_flow")
    raw_flow_path = os.path.join(args.repo_path, "custom_demo", "custom_raw_flow")
    npy_path = os.path.join(args.repo_path, "custom_demo", "custom_npy")

    rawlst = glob(os.path.join(raw_path, "*"))
    rawlst.sort()

    with torch.no_grad():

        # Iterate over all selected images to generate 
        # .npy files and flow graphs, then concatenate 
        # with the original images. 
        for i in range(len(rawlst) - 1):

            img1_path = rawlst[i]
            img2_path = rawlst[i+1]
            image1 = load_image(img1_path)
            image2 = load_image(img2_path)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            save_flow_path = os.path.join(flow_path, os.path.basename(img1_path))
            save_raw_flow_path = os.path.join(raw_flow_path, os.path.basename(img1_path))
            save_npy_path = os.path.join(npy_path, os.path.basename(img1_path.replace(".png", ".npy")))
            
            # 2 channels outputs.
            flow_npy = flow_up[0].permute(1,2,0).cpu().numpy()
            np.save(save_npy_path, flow_npy)
            # 3 channels outputs.
            viz(image2, flow_up, save_flow_path, save_raw_flow_path)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--repo_path', help="path of MotionDetection repository")
    args = parser.parse_args()

    demo(args)