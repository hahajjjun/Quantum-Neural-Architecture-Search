# %%
import numpy as np
import torch
import torch.nn as nn
# %%
class Agent(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_steps=4, device=''):
        super(Agent, self).__init__()

        self.DEVICE = device
        self.num_filter_option = 3
        self.filter_size_option = 3

        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, self.num_filter_option)

        self.num_steps = num_steps
        self.nhid = hidden_size
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype = torch.float, device = self.DEVICE)
        c_t = torch.zeros(1, self.nhid, dtype = torch.float, device = self.DEVICE)
        return (h_t, c_t)

    def forward(self, input):
        outputs = []
        h_t, c_t = self.hidden
        for i in range(self.num_steps):
            h_t, c_t = self.lstm(input, (h_t, c_t))
            output = self.decoder(h_t)
            input = output
            outputs += [output]
        outputs = torch.stack(outputs).squeeze(1)
        return outputs