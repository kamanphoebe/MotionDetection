import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet import FastFlowNet
from flow_vis import flow_to_color
import json
from PIL import Image
import yaml
from nuscenes.nuscenes import NuScenes

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

# Settings
sensor = config["sensor"]
use_visibility = config["use_visibility"]
max_dist = config["max_dist"]
scene_num_train = config["scene_num_train"]
use_scene_path = config["use_scene_path"]
nuscenes_path = config["nuscenes_path"]
flow_path = config["flow_path_fastflowkitti"][2]
raw_flow_path = config["raw_flow_path_fastflowkitti"][2]
npy_path = config["npy_path_fastflowkitti"][2]

with open(use_scene_path, "r") as f:
    use_scene = json.load(f)

nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=False)
scene_count = 0

for scene in nusc.scene:

    if scene["token"] in use_scene[scene_num_train:]:

        scene_count += 1
        img_count = 0

        if scene_count < 15:
            continue

        sample = nusc.get("sample", scene["first_sample_token"])
        curr_sensor_data = nusc.get("sample_data", sample['data'][sensor])
        curr_sensor_token = curr_sensor_data["token"]

        # Iterate over all CAM_FRONT images in current scene (all frames)
        while curr_sensor_token:

            img_count += 1

            curr_sensor_data = nusc.get("sample_data", curr_sensor_token)
            if curr_sensor_data["next"]:
                next_sensor_token = curr_sensor_data["next"]
                next_sensor_data = nusc.get(
                            "sample_data", next_sensor_token)
            for i in range(5):
                next_sensor_token = next_sensor_data["next"]
                if not next_sensor_token:
                    break
                next_sensor_data = nusc.get(
                    "sample_data", next_sensor_token)
            
            if not next_sensor_token:
                break

            img1_path = nusc.get_sample_data_path(curr_sensor_token)
            img2_path = nusc.get_sample_data_path(next_sensor_token)

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

            save_flow_path = flow_path + f"/scene_{scene_count}_img_{img_count}_{curr_sensor_token}.png"
            save_raw_flow_path = raw_flow_path + f"/scene_{scene_count}_img_{img_count}_{curr_sensor_token}.png"
            save_npy_save_path = npy_path + f"/scene_{scene_count}_img_{img_count}_{curr_sensor_token}.npy"
            
            cv2.imwrite(save_flow_path, flow_color)
            np.save(save_npy_save_path, flow)

            # Concatenate the flow graph and its corresponding original image.
            original = Image.open(img2_path)
            flow = Image.open(save_flow_path)
            w = original.width + flow.width
            h = original.height
            concat = Image.new("RGB", (w, h))
            concat.paste(original, (0, 0))
            concat.paste(flow, (original.width, 0))

            concat.save(save_raw_flow_path)

            curr_sensor_token = curr_sensor_data["next"]
