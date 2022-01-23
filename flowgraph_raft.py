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

import json
import yaml
import logging


DEVICE = 'cuda'

# Please modify the path of config.yml
with open("{FILE_PATH}/config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

rawlst_paths = config["rawlst_path"]

logging.getLogger().setLevel(logging.INFO)
logger_shapely = logging.getLogger("shapely")
logger_shapely.setLevel(logging.ERROR)

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

    with torch.no_grad():

        # Iterate over all selected images to generate 
        # .flo files and flow graphs, then concatenate 
        # with the original images. 
        for i, rawlst_path in enumerate(rawlst_paths):

            rawlst = open(rawlst_path, 'r')
            seqlst = json.load(rawlst)
            
            for j, seq in enumerate(seqlst):
                
                # Use for preventing job hanging in bifrost.
                if j % 10 == 9:
                    logging.info(f"Finished seq {j}")

                # Iterate over all selected sequential images.
                for k, curr_img in enumerate(seq[:-1]):

                    next_img = seq[k+1]
                    img1_path = curr_img["imgpath"]
                    img2_path = next_img["imgpath"]
                    image1 = load_image(img1_path)
                    image2 = load_image(img2_path)

                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

                    flow_path = config["flow_train_path_raftkitti"][i] + f"/seq_{j+1}_img_{k+1}_{curr_img['name']}.png"
                    raw_flow_path = config["raw_flow_train_path_raftkitti"][i] + f"/seq_{j+1}_img_{k+1}_{curr_img['name']}.png"
                    npy_path = config["flow_train_path_raftkitti_2"][i] + f"/seq_{j+1}_img_{k+1}_{curr_img['name']}.npy"
                    
                    # 2 channels outputs.
                    flow_npy = flow_up[0].permute(1,2,0).cpu().numpy()
                    np.save(npy_path, flow_npy)
                    # 3 channels outputs.
                    viz(image1, flow_up, flow_path, raw_flow_path)
                
            rawlst.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
