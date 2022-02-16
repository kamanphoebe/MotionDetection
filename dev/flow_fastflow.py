import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet import FastFlowNet
from flow_vis import flow_to_color
import json
from PIL import Image
import yaml

div_flow = 20.0
div_size = 64


def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(
        b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

model = FastFlowNet().cuda().eval()
model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_kitti.pth'))

# Please modify the path of config.yml
with open("{FILE_PATH}/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

rawlst_paths = config["rawlst_path"]
flow_path = config["flow_path_fastflowkitti"][2]
raw_flow_path = config["raw_flow_path_fastflowkitti"][2]
npy_path = config["npy_path_fastflowkitti"][2]


# Iterate over all selected images to generate 
# .npy files and flow graphs, then concatenate 
# with the original images. 
for i, rawlst_path in enumerate(rawlst_paths):

    rawlst = open(rawlst_path, 'r')
    seqlst = json.load(rawlst)

    for j, seq in enumerate(seqlst):

        # Iterate over all selected sequential images.
        for k, curr_img in enumerate(seq[:-1]):

            next_img = seq[k+1]
            img1_path = curr_img["imgpath"]
            img2_path = next_img["imgpath"]

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

            flow_path = flow_path + f"/seq_{j+1}_img_{k+1}_{curr_img['name']}.png"
            raw_flow_path = raw_flow_path + f"/seq_{j+1}_img_{k+1}_{curr_img['name']}.png"
            npy_path = npy_path + f"/seq_{j+1}_img_{k+1}_{curr_img['name']}.npy"
            
            cv2.imwrite(flow_path, flow_color)
            np.save(npy_path, flow)

            # Concatenate the flow graph and its corresponding original image.
            original = Image.open(img2_path)
            flow = Image.open(flow_path)
            w = original.width + flow.width
            h = original.height
            concat = Image.new("RGB", (w, h))
            concat.paste(original, (0, 0))
            concat.paste(flow, (original.width, 0))

            concat.save(raw_flow_path)

    rawlst.close()