from torch.utils.tensorboard import SummaryWriter
from motdataset import MotionDetDataset
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
import yaml
import datetime
import pytz
import warnings
import logging

# Use GPU.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.INFO)
logger_shapely = logging.getLogger("shapely")
logger_shapely.setLevel(logging.ERROR)

with open("./config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

label_paths = config["label_path_fastflowkitti"]
model_path = config["model_path"]
checkpoint_path = config["checkpoint_path"]
writer_path = config["writer_path"]

# Model parameters.
NUM_EPOCHS = config["num_epoch"]
BATCH_SIZE = config["batch_size"]
LEARNING_RATE = config["learning_rate"]
BATCHES_PER_ITER = config["batches_per_iter"]
NUM_WORKERS = config["num_workers"]
MOMENTUM = config["momentum"]
WEIGHT_DECAY = config["weight_decay"]

model = models.resnet18(pretrained=False)
model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
in_ftrs = model.fc.in_features
model.fc = nn.Linear(in_ftrs, out_features=1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)


cn_timezone = pytz.timezone("Asia/Shanghai")
timestamp = datetime.datetime.now(tz=cn_timezone).strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"{writer_path}/run_{timestamp}")


with open(label_paths[0]) as csvf:
    trainset = MotionDetDataset(csvf, datatype="train")

with open(label_paths[1]) as csvf:
    validset = MotionDetDataset(csvf, datatype="valid")

trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

validloader = DataLoader(
    validset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Train an epoch and report scores per iter.
# Return four average scores.
def train_one_epoch(epoch_idx):
    running_loss = 0.0
    running_f1 = 0.0
    running_prec = 0.0
    running_rec = 0.0
    last_loss = 0.0
    last_f1 = 0.0
    last_prec = 0.0
    last_rec = 0.0

    # Iterate over batches.
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        normal = nn.Sigmoid()
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        # Compute F1 score.
        outputs[normal(outputs) > 0.5] = 1
        outputs[normal(outputs) <= 0.5] = 0
        prec, rec, f1, _ = precision_recall_fscore_support(
            labels.cpu().detach().numpy(), outputs.squeeze().cpu().detach().numpy(), average="binary")

        running_loss += loss.item()
        running_f1 += f1
        running_prec += prec
        running_rec += rec
        # Report scores every iter i.e. BATCH_GATHER batches.
        if i % BATCHES_PER_ITER == BATCHES_PER_ITER - 1:
            last_loss = running_loss / BATCHES_PER_ITER
            last_f1 = running_f1 / BATCHES_PER_ITER
            last_prec = running_prec / BATCHES_PER_ITER
            last_rec = running_rec / BATCHES_PER_ITER
            logging.info("\tbatch %d loss: %.3f F1: %.3f" % (i+1, last_loss, last_f1))
            tb_x = epoch_idx * len(trainloader) + i + 1
            writer.add_scalars(
                "Loss&F1/train", {"Loss": last_loss, "F1": last_f1, "Precision": last_prec, "Recall": last_rec}, tb_x)
            running_loss = 0.0
            running_f1 = 0.0
            running_prec = 0.0
            running_rec = 0.0

    return last_loss, last_f1, last_prec, last_rec


# Evaluate an epoch and return four average scores.
def eval_one_epoch():
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
        # Compute F1 score.
        voutputs[normal(voutputs) > 0.5] = 1
        voutputs[normal(voutputs) <= 0.5] = 0
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

    return avg_vloss, avg_vf1, avg_vprec, avg_vrec


# Report scores to stdout, file and tensorboard.
def report_one_epoch(train_scores, eval_scores):
    scores_name = ["Loss", "F1", "Precision", "Recall"]

    # Print to stdout.
    logging.info("LOSS train %.3f valid %.3f" % (train_scores[0], eval_scores[0]))
    logging.info("F1 train %.3f valid %.3f" % (train_scores[1], eval_scores[1]))

    # Print to tensorboard.
    for i in range(len(train_scores)):
        writer.add_scalars(f"Train VS Valid {scores_name[i]}", {
                        "Training": train_scores[i], "Validation": eval_scores[i]}, epoch_idx+1)
    writer.flush()


max_vf1 = 0.0

logging.info("Start training...")
for epoch_idx in range(NUM_EPOCHS):
    logging.info(f"EPOCH {epoch_idx+1}")

    # Scores are returned in the order of 
    # 0=loss, 1=f1, 2=precision, 3=recall
    model.train()
    train_scores = train_one_epoch(epoch_idx)
    model.eval()
    eval_scores = eval_one_epoch()
    report_one_epoch(train_scores, eval_scores)
    scheduler.step()
    

    # Save the best parameters.
    if eval_scores[1] > max_vf1:
        max_vf1 = eval_scores[1]
        save_path = f"{model_path}/model_{timestamp}.pth"
        torch.save(model.state_dict(), save_path)

    # Save checkpoints every 10 epochs.
    if epoch_idx and (epoch_idx + 1) % 10 == 0:
        checkpt_path = f"{checkpoint_path}/checkpt_{timestamp}"
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'max_vf1': max_vf1
        }, checkpt_path)

writer.close()

# %%
