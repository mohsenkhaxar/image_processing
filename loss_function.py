import torch.optim as optim
import torch.nn as nn
from convolutional_neural_network import Net

class LossFunction:

    @staticmethod
    def loss_function():
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        return (criterion, optimizer)