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

with open("./config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Settings.
use_category = config["use_category"]
use_visibility = config["use_visibility"]
min_dist = config["min_dist"]
max_dist = config["max_dist"]
threshold_vel = config["threshold_vel"]

nuscenes_path = config["nuscenes_path"]
flow_paths = config["flow_path_fastflowkitti_2"]
label_paths = config["label_path_fastflowkitti_2"]

# Compute distance between box and ego.
def box_distance(sample_data, ann_data):
    ego_location = nusc.get("ego_pose", sample_data["ego_pose_token"])[
        "translation"]
    box_world_location = ann_data["translation"]
    distance = np.linalg.norm(
        np.array(box_world_location) - np.array(ego_location))
    return distance


nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=False)

for i in range(2):
    
    count = 0
    label_dict = []
    npy_paths = glob.glob(flow_paths[i] + "/*")

    for npy_path in npy_paths:

        reg = re.search(
            rf"{flow_paths[i]}/seq_([0-9]*)_img_([0-9]*)_(.*).npy", npy_path)
        sd_token = reg.group(3)

        sd_data = nusc.get("sample_data", sd_token)
        _, curr_boxlst, _ = nusc.get_sample_data(
            sd_token, box_vis_level=BoxVisibility.ALL)

        for cbox in curr_boxlst:

            if cbox.name not in use_category:
                continue
            annotation = nusc.get("sample_annotation", cbox.token)
            visibility_token = annotation["visibility_token"]
            if nusc.get("visibility", visibility_token)["level"] != use_visibility:
                continue
            distance = box_distance(sd_data, annotation)
            if distance >= min_dist and distance <= max_dist:
                curr_instance_token = annotation["instance_token"]
                next_sd_token = sd_data["next"]
                next_sd_data = nusc.get("sample_data", next_sd_token)
                while not next_sd_data["is_key_frame"]:
                    next_sd_token = next_sd_data["next"]
                    if not next_sd_token:
                        break
                    next_sd_data = nusc.get("sample_data", next_sd_token)
                if not next_sd_data:
                    break

                _, next_boxlst, _ = nusc.get_sample_data(
                    next_sd_token, box_vis_level=BoxVisibility.ALL)

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
                        if next_distance >= min_dist and next_distance <= max_dist:
                            cs_data = nusc.get('calibrated_sensor', sd_data['calibrated_sensor_token'])
                            cam_intrinsic = np.array(cs_data["camera_intrinsic"])
                            corners = view_points(
                                cbox.corners(), view=cam_intrinsic, normalize=True)[:2, :]
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

                            velocity = np.linalg.norm(
                                box_velocity(nusc, cbox.token, nbox.token))
                            motionFlag = 1 if velocity >= threshold_vel else 0
                            label_dict.append(dict())
                            label_dict[-1]["npypath"] = npy_path
                            label_dict[-1]["xmin"] = xmin
                            label_dict[-1]["xmax"] = xmax
                            label_dict[-1]["ymin"] = ymin
                            label_dict[-1]["ymax"] = ymax
                            label_dict[-1]["motionFlag"] = motionFlag


    # Outut csv file.
    f = open(label_paths[i], 'w')

    csvheaders = ["npypath", "xmin", "xmax", "ymin", "ymax", "motionFlag"]
    writer = csv.DictWriter(f, fieldnames=csvheaders)
    writer.writeheader()
    writer.writerows(label_dict)

    f.close()
