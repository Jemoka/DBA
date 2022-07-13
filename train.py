# topic modeling
from gensim import downloader
from pandas._libs.lib import infer_datetimelike_array

# torch
import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F

# wandb
import wandb

# coreutils
import os, glob

# pandas and numpy
import pandas as pd
import numpy as np

# re
import re

# random utils
import random

# import tqdm
from tqdm import tqdm

# import our model
from model import Model

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model
CONFIG = {
    "topicmodeling_model": "glove-wiki-gigaword-50",
    "topicmodeling_size": 50,
    "epochs": 3,
    "lr": 3e-3,
    "batch_size": 2
}

# set up the run
# run = wandb.init(project="DBA", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBA", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

# load dataset dir
DATASET = "./data/wordinfo/pitt-07-13" # TBD

# calculate other dirs
control_dir = os.path.join(DATASET, "control/*.csv")
dementia_dir = os.path.join(DATASET, "dementia/*.csv")
CONTROL_FILES = glob.glob(control_dir)
DEMENTIA_FILES = glob.glob(dementia_dir)

# validation samples
VAL_SAMPLES = 5



#### Data Manipulation ####
# control data
control_dfs = []
# for each file, append and remove nonalphaneumeric
for sample in CONTROL_FILES:
    df = pd.read_csv(sample, index_col=0)
    df["word"] = df["word"].apply((lambda x:''.join(e for e in str(x) if e.isalnum())), 1)
    control_dfs.append(df)

# control data
dementia_dfs = []
# for each file, append and remove nonalphaneumeric
for sample in DEMENTIA_FILES:
    df = pd.read_csv(sample, index_col=0)
    df["word"] = df["word"].apply((lambda x:''.join(e for e in str(x) if e.isalnum())), 1)
    dementia_dfs.append(df)

# shuffle data
random.shuffle(control_dfs)
random.shuffle(dementia_dfs)

# cut data
cut_length = min(len(control_dfs), len(dementia_dfs))
control_dfs = control_dfs[:cut_length]
dementia_dfs = dementia_dfs[:cut_length]

# holdout
control_train = control_dfs[:-VAL_SAMPLES]
dementia_train = dementia_dfs[:-VAL_SAMPLES]

control_test = control_dfs[-VAL_SAMPLES:]
dementia_test = dementia_dfs[-VAL_SAMPLES:]

# combine and shuffle
control_train = list(zip(control_train, [0 for _ in range(len(control_train))]))
dementia_train = list(zip(dementia_train, [1 for _ in range(len(dementia_train))]))

control_test = list(zip(control_test, [0 for _ in range(len(control_test))]))
dementia_test = list(zip(dementia_test, [1 for _ in range(len(dementia_test))]))

train_data = control_train+dementia_train
test_data = control_test+dementia_test

random.shuffle(train_data)
random.shuffle(test_data)



#### 
# load a word2vec model
wv_model = downloader.load(config.topicmodeling_model)

# load the model and optimizer
model = Model().to(DEVICE)
optim = AdamW(model.parameters(), lr=config.lr)

### training utilities ###
# define validation tools
def eval_model_on_batch(model_output_encoded, targets):
    # calculate accuracy, precision, recall
    # calculate pos/neg/etc.
    true_pos = torch.logical_and(model_output_encoded == targets,
                                model_output_encoded.bool())
    true_neg = torch.logical_and(model_output_encoded == targets,
                                torch.logical_not(model_output_encoded.bool()))
    false_pos = torch.logical_and(model_output_encoded != targets,
                                model_output_encoded.bool())
    false_neg = torch.logical_and(model_output_encoded != targets,
                                torch.logical_not(model_output_encoded.bool()))

    # create the counts
    true_pos = torch.sum(true_pos).cpu().item()
    true_neg = torch.sum(true_neg).cpu().item()
    false_pos = torch.sum(false_pos).cpu().item()
    false_neg = torch.sum(false_neg).cpu().item()

    acc = (true_pos+true_neg)/len(targets)
    if (true_pos+false_pos) == 0:
        prec = 0
    else:
        prec = true_pos/(true_pos+false_pos)

    if (true_pos+false_neg) == 0:
        recc = 0
    else:
        recc = true_pos/(true_pos+false_neg)

    # and return
    return acc, prec, recc

# process a single sample
def process_sample(sample, wv_model):
    # define sample
    x = sample

    # normalize time to a minute to make normal
    time_norm = np.array([x["start"], x["end"]])/60000

    # try to get vec
    try:
        vec = wv_model[str(x["word"]).lower()]
    except KeyError:
        vec = np.full(config.topicmodeling_size, -1)

    # concat and return
    return np.concatenate((time_norm, vec))

# prep data
def pack_prep_data(in_data, wv_model):
    # create a sequence
    in_sec = []

    # for each element
    for elem in in_data:
        # we apply the elements together and divide times by 60000 to turn ms into minutes
        # for normalization
        try: 
            # take an in sample and apply concat
            in_sample = elem.apply(lambda x:process_sample(x, wv_model), axis=1)
        except KeyError:
            # we ignore OOV samples
            continue

        # convert to numpy arary
        in_sample = torch.Tensor(np.array(in_sample.to_list()))
        # append input data
        in_sec.append(in_sample)

    # create packed element with variable length
    in_packed = pack_sequence(in_sec, enforce_sorted=False)

    # return result
    return in_packed

# go through epochs
for e in range(config.epochs):

    print(f"Training epoch {e}...")

    # for each batch
    for batch_id, i in enumerate(tqdm(range(0, len(train_data)-config.batch_size, config.batch_size))):

        batch = train_data[i:i+config.batch_size]

        # seperate in and out data
        in_data,out_data = zip(*batch)

        # create input
        in_packed = pack_prep_data(in_data, wv_model).to(DEVICE)

        # create output
        target_tensor = torch.tensor(out_data).to(DEVICE)
        labels_encoded = F.one_hot(target_tensor, num_classes=2).float()

        # pass to model
        model_out = model(in_packed, labels=labels_encoded)

        # backprop
        model_out["loss"].backward()
        optim.step()
        optim.zero_grad()

        # calculate the accuracy
        model_output_encoded = model_out["logits"].detach().argmax(dim=1)
        acc = torch.sum(model_output_encoded.bool() == target_tensor)/len(target_tensor)

        # plotting to training graph
        run.log({
            "loss": model_out["loss"].cpu().item(),
            "acc": acc.cpu().item()
        })

        # for every 10 batches, randomly perform a single validation sample
        if batch_id % 10 == 0:
            # get the batch
            val_batch = random.randint(0, (len(test_data)//config.batch_size)-1)
            val_batch = test_data[val_batch*config.batch_size:val_batch*config.batch_size + config.batch_size]

            # seperate in and out data
            val_in_data,val_out_data = zip(*val_batch)

            # create input
            val_in_packed = pack_prep_data(val_in_data, wv_model).to(DEVICE)

            # create output
            val_target_tensor = torch.tensor(val_out_data).to(DEVICE)

            # pass to model
            model_out = model(val_in_packed)
            model_output_encoded = torch.argmax(model_out["logits"], axis=1)

            # eval
            acc, prec, recc = eval_model_on_batch(model_output_encoded, val_target_tensor)

            # plot
            run.log({
                "val_accuracy": acc,
                "val_prec": prec,
                "val_recc": recc,
            })

# save model
torch.save(model, f"./models/{run.name}")

