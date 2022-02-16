from nuscenes_patch import box_velocity
from nuscenes_patch import render2d_box
import numpy as np
import glob
import re
import csv
import matplotlib.pyplot as plt
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import view_points
import yaml

with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Settings.
use_category = config["use_category_nuscenes"]
threshold_vel = config["threshold_vel"]
use_visibility = config["use_visibility"]
max_dist = config["max_dist"]

nuscenes_path = config["nuscenes_path"]
flow_path = config["flow_path_fastflowkitti"][2]
label_path = config["label_path_fastflowkitti"][2]


def sort_path(path):
    reg = re.search(
        rf"{flow_path}/scene_([0-9]*)_img_([0-9]*)_(.*).npy", path)
    return (int(reg.group(1)), int(reg.group(2)))


def sort_label(label):
    npy_path = label["npypath"]
    reg = re.search(
        rf"{flow_path}/scene_([0-9]*)_img_([0-9]*)_(.*).npy", npy_path)
    return (int(reg.group(1)), int(reg.group(2)))


# Compute distance between box and ego.
def box_distance(sample_data, ann_data):
    ego_location = nusc.get("ego_pose", sample_data["ego_pose_token"])[
        "translation"]
    box_world_location = ann_data["translation"]
    distance = np.linalg.norm(
        np.array(box_world_location) - np.array(ego_location))
    return distance


def get_2dcorner(box, cam_intrinsic):
    corners = view_points(
        box.corners(), view=cam_intrinsic, normalize=True)[:2, :]
    xmin = corners[0, 0]
    xmax = corners[0, 0]
    ymin = corners[1, 0]
    ymax = corners[1, 0]
    
    for x in corners[0, :]:
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
    for y in corners[1, :]:
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    
    return xmin, xmax, ymin, ymax


nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=False)

count = 0
label_dict = []
npy_paths = glob.glob(flow_path + "/*")
npy_paths.sort(key=sort_path)

for i, npy_path in enumerate(npy_paths):

    reg = re.search(
        rf"{flow_path}/scene_([0-9]*)_img_([0-9]*)_(.*).npy", npy_path)
    if int(reg.group(1)) > 36:
        break
    curr_sd_token = reg.group(3)
    curr_sd_data = nusc.get("sample_data", curr_sd_token)
    if not curr_sd_data["is_key_frame"]:
        continue
    curr_scene = int(reg.group(1))
    _, curr_boxlst, _ = nusc.get_sample_data(
        curr_sd_token, box_vis_level=BoxVisibility.ALL)
    
    # Find the next keyframe.
    for next_npy_path in npy_paths[i+1:]:
        reg = re.search(
            rf"{flow_path}/scene_([0-9]*)_img_([0-9]*)_(.*).npy", next_npy_path)
        next_scene = int(reg.group(1))
        # No keyframe in current scene any more.
        if next_scene != curr_scene:
            break
        next_sd_token = reg.group(3)
        next_sd_data = nusc.get("sample_data", next_sd_token)
        if not next_sd_data["is_key_frame"]:
            continue
        _, next_boxlst, _ = nusc.get_sample_data(
            next_sd_token, box_vis_level=BoxVisibility.ALL)
        next_idx = npy_paths.index(next_npy_path)
        break
    
    if next_scene != curr_scene:
        continue
    # Last frame.
    if next_npy_path == npy_paths[-1] and not next_sd_data["is_key_frame"]:
        break

    for cbox in curr_boxlst:

        if cbox.name not in use_category:
            continue
        annotation = nusc.get("sample_annotation", cbox.token)
        visibility_token = annotation["visibility_token"]
        if nusc.get("visibility", visibility_token)["level"] != use_visibility:
            continue
        distance = box_distance(curr_sd_data, annotation)
        if distance <= max_dist:
            curr_instance_token = annotation["instance_token"]

            for nbox in next_boxlst:
                next_annotation = nusc.get("sample_annotation", nbox.token)
                next_instance_token = next_annotation["instance_token"]

                # Same obj in two frames.
                if next_instance_token == curr_instance_token:
                    next_visibility_token = next_annotation["visibility_token"]
                    if nusc.get("visibility", next_visibility_token)["level"] != use_visibility:
                        break
                    next_sd_data = nusc.get("sample_data", next_sd_token)
                    next_distance = box_distance(next_sd_data, next_annotation)
                    if next_distance <= max_dist:
                        cs_data = nusc.get('calibrated_sensor', curr_sd_data['calibrated_sensor_token'])
                        cam_intrinsic = np.array(cs_data["camera_intrinsic"])
                        cxmin, cxmax, cymin, cymax = get_2dcorner(cbox, cam_intrinsic)
                        nxmin, nxmax, nymin, nymax = get_2dcorner(nbox, cam_intrinsic)
                        xmin_interval = (nxmin - cxmin) / (next_idx - i)
                        xmax_interval = (nxmax - cxmax) / (next_idx - i)
                        ymin_interval = (nymin - cymin) / (next_idx - i)
                        ymax_interval = (nymax - cymax) / (next_idx - i)

                        velocity = np.linalg.norm(
                            box_velocity(nusc, cbox.token, nbox.token))
                        motionFlag = 1 if velocity >= threshold_vel else 0
                        label_dict.append(dict())
                        label_dict[-1]["npypath"] = npy_path
                        label_dict[-1]["xmin"] = cxmin
                        label_dict[-1]["xmax"] = cxmax
                        label_dict[-1]["ymin"] = cymin
                        label_dict[-1]["ymax"] = cymax
                        label_dict[-1]["motionFlag"] = motionFlag

                        for j, middle_npy_path in enumerate(npy_paths[i+1:next_idx], start=1):
                            label_dict.append(dict())
                            label_dict[-1]["npypath"] = middle_npy_path
                            label_dict[-1]["xmin"] = cxmin + xmin_interval * j
                            label_dict[-1]["xmax"] = cxmax + xmax_interval * j
                            label_dict[-1]["ymin"] = cymin + ymin_interval * j
                            label_dict[-1]["ymax"] = cymax + ymax_interval * j
                            label_dict[-1]["motionFlag"] = motionFlag


# Outut csv file.
label_dict.sort(key=sort_label)
with open(label_path, 'w') as f:
    csvheaders = ["npypath", "xmin", "xmax", "ymin", "ymax", "motionFlag"]
    writer = csv.DictWriter(f, fieldnames=csvheaders)
    writer.writeheader()
    writer.writerows(label_dict)
