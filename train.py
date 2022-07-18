# topic modeling
from gensim import downloader
from pandas._libs.lib import infer_datetimelike_array
from sklearn.utils import shuffle

# torch
import torch
from torch.optim import AdamW
from torch.nn.utils.rnn import pack_sequence
import torch.nn.functional as F
from torch.nn import Module, Linear, Flatten

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

# pickling utilities
import pickle

# initialize the device
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize the model
CONFIG = {
    "epochs": 10,
    "lr": 5e-3,
    "batch_size": 16
}

# set up the run
# run = wandb.init(project="DBA", entity="jemoka", config=CONFIG)
run = wandb.init(project="DBA", entity="jemoka", config=CONFIG, mode="disabled")

# get the configuration
config = run.config

# load dataset dir
DATASET = "./data/wordinfo/pitt-07-18-syntax.bin"

# validation samples
VAL_SAMPLES = 0.1

# load the dataset
with open(DATASET, 'rb') as df:
    data = pickle.load(df)
random.shuffle(data)

# split the dataste
train_data = data[:-int(len(data)*VAL_SAMPLES)]
test_data = data[-int(len(data)*VAL_SAMPLES):]

# tensorify the data
train_data_in = torch.tensor(np.array([i[0].todense().squeeze() for i in train_data]))
train_data_out = torch.tensor(np.array([[i[3]] for i in train_data]))

test_data_in = torch.tensor(np.array([i[0].todense().squeeze() for i in test_data]))
test_data_out = torch.tensor(np.array([[i[3]] for i in test_data]))

# define the model
class Model(Module):
    # initialize the model
    def __init__(self, in_dim=1145, out_dim=1):
        # super init
        super().__init__()
        # construct network
        self.d1 = Linear(in_dim, 1024)
        self.d2 = Linear(1024, 128)
        self.d3 = Linear(128, 32)
        self.d4 = Linear(32, out_dim)

    # pass through the network
    def forward(self, x, label=None):
        net = F.relu(self.d1(x))
        net = F.relu(self.d2(net))
        net = torch.tanh(self.d3(net))
        net = torch.tanh(self.d4(net))*3 # we multiply to scale the tanh
                                         # to account for the range of outputs

        if label is not None:
            return {"logit": net,
                    "loss": ((net-label)**2).mean()}
        else:
            return {"logit": net}
        
# define model and optimiers
model = Model().to(DEVICE)
optim = AdamW(model.parameters(), lr=config.lr)

# watch
run.watch(model)
    
for e in range(config.epochs):
    print(f"Training epoch {e}...")

    for indx, batch_id in tqdm(enumerate(range(0, len(train_data)-config.batch_size, config.batch_size)), total=((len(train_data)-config.batch_size)//config.batch_size)):
        # get in and out batches
        batch_in = train_data_in[batch_id:batch_id+config.batch_size].to(DEVICE).float()
        batch_out = train_data_out[batch_id:batch_id+config.batch_size].to(DEVICE).float()

        # pass the data through model
        output = model(batch_in, label=batch_out)

        # backprop the erorr
        output["loss"].backward()
        optim.step()
        optim.zero_grad()

        # log the loss and logits
        run.log({
            "loss": output["loss"].detach().item(),
            "logit": output["logit"].squeeze().detach().cpu()
        })

        # run a sample of validation, ever 10 items
        if indx % 10 == 0:
            # sample a random batch
            val_batch_id = random.randint(0, len(test_data)//config.batch_size)*config.batch_size
            # get in and out samples
            val_in = test_data_in[val_batch_id:val_batch_id+config.batch_size].to(DEVICE).float()
            val_out = test_data_out[val_batch_id:val_batch_id+config.batch_size].to(DEVICE).float()
            # pass in and calculate error
            output = model(val_in, label=val_out)
            # log the loss and logits
            run.log({
                "val_loss": output["loss"].detach().item(),
                "val_logit": output["logit"].squeeze().detach().cpu()
            })





