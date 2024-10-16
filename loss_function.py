import torch.optim as optim
import torch.nn as nn
from image_processing.convolutional_neural_network import Net

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)