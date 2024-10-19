import torch
from loading_cifar10 import LoadCifar10
from convolutional_neural_network import Net


class NetOnAllTestData:

    @staticmethod
    def net_on_all_testdata():
        # load trained model
        PATH = './cifar_net.pth'
        net = Net()
        net.load_state_dict(torch.load(PATH, weights_only=True))
        print('model loaded')

        # calculate net accuracy on whole test data
        _, _, _, testloader = LoadCifar10.load_cifar10()
        correct = 0
        total = 0

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'accuracy of the network on 10000 test images: , {100 * correct // total} %')

# NetOnAllTestData.net_on_all_testdata()
