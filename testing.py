import torch
from torchvision.utils import make_grid
from loading_cifar10 import LoadCifar10
from showing_images import ShowImages
from convolutional_neural_network import Net


class Test:

    @staticmethod
    def test():
        batch_size, classes, _, testloader = LoadCifar10.load_cifar10()

        # load trained model
        net = Net()
        PATH = '.cifar_net.pth'
        net.load_state_dict(torch.load(PATH, weights_only=True))
        print('model loaded')

        # show images from test set
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        ShowImages().imgshow(make_grid(images))
        print('ground truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

        # show predicted images on test set
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        print('predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

Test.test()