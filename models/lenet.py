'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        

        self.classifier=nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(inplace=True),#cluster_activation(num_clusters=512),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),#cluster_activation(num_clusters=512),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        #print 'x before flattening:',x.size()
        x = x.view(x.size(0), -1)
        #print 'x after flattening:',x.size()
        x = self.classifier(x)



        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        return x
