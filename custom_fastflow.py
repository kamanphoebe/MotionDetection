import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet import FastFlowNet
from flow_vis import flow_to_color
from PIL import Image
from glob import glob
import os
import argparse

div_flow = 20.0
div_size = 64


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(
        b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_kitti.pth'))


def visual(args):

    raw_path = os.path.join(args.path, "custom_demo", "custom_raw")
    flow_path = os.path.join(args.path, "custom_demo", "custom_flow")
    raw_flow_path = os.path.join(args.path, "custom_demo", "custom_raw_flow")
    npy_path = os.path.join(args.path, "custom_demo", "custom_npy")

    rawlst = glob(os.path.join(raw_path, "*"))
    rawlst.sort()

    # Iterate over all selected images to generate 
    # .npy files and flow graphs, then concatenate 
    # with the original images. 
    for i in range(len(rawlst) - 1):

        img1_path = rawlst[i]
        img2_path = rawlst[i+1]

        img1 = torch.from_numpy(cv2.imread(img1_path)).float().permute(
            2, 0, 1).unsqueeze(0)/255.0
        img2 = torch.from_numpy(cv2.imread(img2_path)).float().permute(
            2, 0, 1).unsqueeze(0)/255.0
        img1, img2, _ = centralize(img1, img2)

        height, width = img1.shape[-2:]
        orig_size = (int(height), int(width))

        if height % div_size != 0 or width % div_size != 0:
            input_size = (
                int(div_size * np.ceil(height / div_size)),
                int(div_size * np.ceil(width / div_size))
            )
            img1 = F.interpolate(img1, size=input_size,
                                    mode='bilinear', align_corners=False)
            img2 = F.interpolate(img2, size=input_size,
                                    mode='bilinear', align_corners=False)
        else:
            input_size = orig_size

        input_t = torch.cat([img1, img2], 1).cuda()

        output = model(input_t).data

        flow = div_flow * \
            F.interpolate(output, size=input_size,
                            mode='bilinear', align_corners=False)

        if input_size != orig_size:
            scale_h = orig_size[0] / input_size[0]
            scale_w = orig_size[1] / input_size[1]
            flow = F.interpolate(flow, size=orig_size,
                                    mode='bilinear', align_corners=False)
            flow[:, 0, :, :] *= scale_w
            flow[:, 1, :, :] *= scale_h

        flow = flow[0].cpu().permute(1, 2, 0).numpy()

        flow_color = flow_to_color(flow, convert_to_bgr=True)

        save_flow_path = os.path.join(flow_path, os.path.basename(img1_path))
        save_raw_flow_path = os.path.join(raw_flow_path, os.path.basename(img1_path))
        save_npy_path = os.path.join(npy_path, os.path.basename(img1_path.replace(".png", ".npy")))
        
        cv2.imwrite(save_flow_path, flow_color)
        np.save(save_npy_path, flow)

        # Concatenate the flow graph and its corresponding original image.
        original = Image.open(img2_path)
        flow = Image.open(save_flow_path)
        w = original.width + flow.width
        h = original.height
        concat = Image.new("RGB", (w, h))
        concat.paste(original, (0, 0))
        concat.paste(flow, (original.width, 0))

        concat.save(save_raw_flow_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help="path of MotionDetection repository")
    args = parser.parse_args()

    visual(args)