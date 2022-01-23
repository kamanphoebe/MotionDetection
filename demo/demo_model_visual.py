from motdataset import MotionDetDataset
from nuscenes_patch import render2d_box
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support
import yaml
import csv
from PIL import Image
import matplotlib.pyplot as plt
import re
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.getLogger().setLevel(logging.INFO)
logger_shapely = logging.getLogger("shapely")
logger_shapely.setLevel(logging.ERROR)

with open("../config.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

BATCH_SIZE = config["batch_size"]
NUM_WORKERS = config["num_workers"]
model_path = config["model_path"]
demo_visual_path = config["demo_visual_path"]
demo_raw_path = config["demo_raw_path"]
demo_label_path = config["demo_label_path"]
sampling_frac = config["sampling_frac_valid"]

with open(demo_label_path) as csvf:
    labels = list(csv.DictReader(csvf))

with open(demo_label_path) as csvf:
    validset = MotionDetDataset(csvf, datatype="infer")

validloader = DataLoader(
    validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

model = models.resnet18()
model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
in_ftrs = model.fc.in_features
model.fc = nn.Linear(in_ftrs, out_features=1)
model.load_state_dict(torch.load(f"{model_path}/model_fastflow.pth"))
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()


# Inference using the loaded model.
# Return a list containing all outputs.
def infer():

    model.eval()
    con_results = []
    bi_results = []

    running_vloss = 0.0
    running_vf1 = 0.0
    running_vprec = 0.0
    running_vrec = 0.0

    for i, vdata in enumerate(validloader):
        vinputs, vlabels = vdata
        vinputs, vlabels = vinputs.to(device), vlabels.to(device)
        voutputs = model(vinputs)
        normal = nn.Sigmoid()
        vloss = criterion(voutputs.squeeze(), vlabels.float())
        con_results += normal(voutputs).tolist()
        # Binarization.
        voutputs[normal(voutputs) > 0.5] = 1
        voutputs[normal(voutputs) <= 0.5] = 0
        bi_results += voutputs.tolist()
        # Compute F1 score.
        vprec, vrecall, vf1, _ = precision_recall_fscore_support(
            vlabels.cpu().detach().numpy(), voutputs.squeeze().cpu().detach().numpy(), average="binary")
        running_vloss += vloss.item()
        running_vf1 += vf1
        running_vprec += vprec
        running_vrec += vrecall

    avg_vloss = running_vloss / (i + 1)
    avg_vf1 = running_vf1 / (i + 1)
    avg_vprec = running_vprec / (i + 1)
    avg_vrec = running_vrec / (i + 1)

    logging.info("\nResults of inference:")
    logging.info(f"\tloss: {avg_vloss}\n\tF1: {avg_vf1}\n\tprec: {avg_vprec}\n\trec:{avg_vrec}")

    return con_results, bi_results


if __name__ == "__main__":

    con_results, bi_results = infer()
    fig = plt.figure(figsize=(42.5, 11.8))

    for i, label in enumerate(labels):

        curr_imgpath = label["npypath"].replace(".npy", ".png")
        curr_imgpath = curr_imgpath.replace(
            "demo_flow", "demo_raw")

        if i == 0:
            prev_imgpath = curr_imgpath

        # If all boxes in current image have already rendered,
        # then output the image.
        if curr_imgpath != prev_imgpath:
            reg = re.search(rf"{demo_raw_path}/(.+).png", prev_imgpath)
            imgname = reg.group(1)
            reg = re.search(r"seq_([0-9]*)_img_(.+)", imgname)
            idx = int((int(reg.group(1)) - 1) / 20)
            plt.savefig(
                f"{demo_visual_path}/{imgname}.png", bbox_inches="tight", pad_inches=0)
            plt.clf()

        img = Image.open(curr_imgpath)
        ax = plt.subplot(111)
        ax.set_axis_off()
        ax.imshow(img)

        # Render boxes in current image.
        xmin = float(label["xmin"])
        xmax = float(label["xmax"])
        ymin = float(label["ymin"])
        ymax = float(label["ymax"])
        width = 1600
        if bi_results[i][0] == 1:
            linecolor = "red"
            textcolor = "red"
            linestyle = "solid"
        else:
            linecolor = "dodgerblue"
            textcolor = "dodgerblue"
            linestyle = "dashed"
        # Use another linecolor if output is incorrect.
        if bi_results[i][0] != int(label["motionFlag"]):
            linecolor = "fuchsia"
        text = "{:.3f}".format(con_results[i][0])
        # Render boxes in both raw image and flow graph.
        rect_in_raw = np.array(
            [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        rect_in_flow = np.array(
            [[xmin+width, ymin], [xmax+width, ymin], [xmax+width, ymax], [xmin+width, ymax]])
        render2d_box(axis=ax, corners=rect_in_raw,
                      linecolor=linecolor, linestyle=linestyle, text=text, textcolor=textcolor)
        render2d_box(axis=ax, corners=rect_in_flow,
                      linecolor=linecolor, linestyle=linestyle, text=text, textcolor=textcolor)

        prev_imgpath = curr_imgpath
    
    logging.info("Finished processing all labels.")
