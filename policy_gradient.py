# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.functional import one_hot, log_softmax, softmax, normalize
from torch.distributions import Categorical
from collections import deque

from controller import Agent
from model import NASModel
# %%
from torch.utils.tensorboard import SummaryWriter
# %%
class PolicyGradient:
    def __init__(self, config, train_set, test_set, use_cuda=False):
        self.NUM_EPOCHS = config.NUM_EPOCHS
        self.ALPHA = config.ALPHA
        self.BATCH_SIZE = config.BATCH_SIZE
        self.HIDDEN_SIZE = config.HIDDEN_SIZE
        self.BETA = config.BETA
        self.GAMMA = config.GAMMA
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.INPUT_SIZE = config.INPUT_SIZE
        self.NUM_STEPS = config.NUM_STEPS
        self.ACTION_SPACE = config.ACTION_SPACE

        self.train = train_set
        self.test = test_set

        self.writer = SummaryWriter(comment=f'_PG_CP_Gamme={self.GAMMA},'
                                            f'LR={self.ALPHA},'
                                            f'BS={self.BATCH_SIZE},'
                                            f'NH={self.HIDDEN_SIZE},'
                                            f'BETA={self.BETA}')
        if use_cuda:
            self.agent = Agent(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_STEPS, device = self.DEVICE).cuda()
        else:
            self.agent = Agent(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_STEPS, device=self.DEVICE)
        self.adam = optim.Adam(params=self.agent.parameters(), lr = self.ALPHA)
        self.total_rewards = deque([], maxlen = 100)
    
    def play_episode(self):

        init_state = [[3,8,16]]
        episode_logits = self.agent(torch.tensor(init_state).float().to(self.DEVICE))
        action_index = Categorical(logits=episode_logits).sample().unsqueeze(1)
        mask = one_hot(action_index, num_classes = self.ACTION_SPACE)
        episode_log_probs = torch.sum(mask.float()*log_softmax(episode_logits, dim=1), dim=1)
        action_space = torch.tensor([[3,5,7],[8,16,32],[3,5,7],[8,16,32]], device = self.DEVICE)
        action = torch.gather(action_space, 1, action_index)
        
        net = NASModel(action)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum =0.9)

        for epoch in range(2):
            running_loss = 0.0
            for i, data in enumerate(self.train, 0):
                inputs,labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

        # Evaluate
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: {}'.format(acc))

        # compute the reward
        reward = acc

        episode_weighted_log_probs = episode_log_probs * reward
        sum_weighted_log_probs = torch.sum(episode_weighted_log_probs).unsqueeze(dim=0)

        return  sum_weighted_log_probs, episode_logits, reward
    
    def calculate_loss(self, epoch_logits: torch.Tensor, weighted_log_probs: torch.Tensor):
        policy_loss = -1 * torch.mean(weighted_log_probs)

        # add the entropy bonus
        p = softmax(epoch_logits, dim=1)
        log_p = log_softmax(epoch_logits, dim=1)
        entropy = -1 * torch.mean(torch.sum(p * log_p, dim=1), dim=0)
        entropy_bonus = -1 * self.BETA * entropy

        return policy_loss + entropy_bonus, entropy

    def solve_environment(self):
        epoch = 0
        while epoch < self.NUM_EPOCHS:
            epoch_logits = torch.empty(size=(0,self.ACTION_SPACE) , device = self.DEVICE)
            epoch_weighted_log_probs = torch.empty(size=(0,), dtype = torch.float, device = self.DEVICE)

            for i in range(self.BATCH_SIZE):
                (
                    episode_weighted_log_prob_trajectory,
                    episode_logits,
                    sum_of_episode_rewards
                ) = self.play_episode()
                self.total_rewards.append(sum_of_episode_rewards)
                epoch_weighted_log_probs = torch.cat((epoch_weighted_log_probs, episode_weighted_log_prob_trajectory),dim=0)
                epoch_logits = torch.cat((epoch_logits, episode_logits), dim=0)
            loss, entropy = self.calculate_loss(epoch_logits=epoch_logits,weighted_log_probs=epoch_weighted_log_probs)
            self.adam.zero_grad()
            loss.backward()
            self.adam.step()

            print("\r", f"Epoch: {epoch}, Avg Return per Epoch: {np.mean(self.total_rewards):.3f}",
                  end="",
                  flush=True)
            self.writer.add_scalar(tag='Average Return over 100 episodes',
                                   scalar_value=np.mean(self.total_rewards),
                                   global_step=epoch)
            self.writer.add_scalar(tag='Entropy',
                                   scalar_value=entropy,
                                   global_step=epoch)
            epoch += 1
        self.writer.close()