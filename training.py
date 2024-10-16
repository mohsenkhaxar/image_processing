from image_processing.loading_cifar10 import trainloader
from image_processing.loss_function import optimizer, criterion
from image_processing.convolutional_neural_network import Net
import torch

net = Net()
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss = loss.item()
        if i % 2000 == 1999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('finished training')

# save trained model
PATH = '.cifar_net.pth'
torch.save(net.state_dict(), PATH)
print('model saved')
