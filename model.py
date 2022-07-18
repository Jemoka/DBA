# import torch
import torch

# neural network utilities
from torch.nn import LSTM, Linear, Module, BCEWithLogitsLoss
import torch.nn.functional as F

# new model!
class Model(Module):

    def __init__(self, num_layers=2, hidden_dim=128, dropout=0.1, meta_size=2, vocab_size=50, out_size=2):
        # initalize
        super().__init__()

        # lstm
        self.lstm = LSTM(meta_size+vocab_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)

        # output
        self.out = Linear(hidden_dim, out_size)

        # loss
        self.bce_loss = BCEWithLogitsLoss(pos_weight=torch.ones([out_size]))

    # forward
    def forward(self, batch, labels=None):
        # pass through LSTM
        _, (h_n, _) = self.lstm(batch)

        # get final hidden layer
        final_hidden = h_n[-1]

        # pass to output
        out = self.out(final_hidden)

        # if labels, calculate loss
        if labels is not None:
            loss = self.bce_loss(out, labels)
            return {"logits": out, "loss": loss}
        else:
            return {"logits": out}

