from loading_cifar10 import LoadCifar10
from loss_function import LossFunction
from convolutional_neural_network import Net
import torch


class Train:

    @staticmethod
    def train():

        net = Net()
        for epoch in range(2):
            running_loss = 0.0
            _, _, trainloader, _ = LoadCifar10.load_cifar10()
            for i, data in enumerate(trainloader, 0):

                criterion, optimizer = LossFunction.loss_function()

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

        print('training finished')

        # save trained model
        PATH = '.cifar_net.pth'
        torch.save(net.state_dict(), PATH)
        print('model saved')

# Train.train()
