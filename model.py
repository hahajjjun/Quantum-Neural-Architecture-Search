# %%
import torch.nn as nn
import torch.nn.functional as F
# %%
class NASModel(nn.Module):
    def __init__(self, actions):
        super(NASModel, self).__init__()
        self.kernel_1, self.filters_1, self.kernel_2, self.filters_2 = actions.tolist()
        self.kernel_1 = self.kernel_1[0]
        self.filters_1 = self.filters_1[0]
        self.kernel_2 = self.kernel_2[0]
        self.filters_2 = self.filters_2[0]
        self.conv1 = nn.Conv2d(1, self.filters_1, self.kernel_1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(self.filters_1, self.filters_2, self.kernel_2)
        self.tmp = int(self.filters_2 * ((29-self.kernel_1)/2 - self.kernel_2 + 1)**2)
        self.fc1 = nn.Linear(self.tmp, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.tmp)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# %%
# class QASModel(nn.Module):
