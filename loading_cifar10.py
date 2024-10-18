import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


class LoadCifar10:

    @staticmethod
    def load_cifar10():
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

        batch_size = 4

        trainset = CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transform,
        )

        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=0,
        )

        testset = CIFAR10(
            root='./data',
            train=False,
            download=False,
            transform=transform,
        )

        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0,
        )

        classes = (
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        )

        return (batch_size, classes, trainloader, testloader)

# LoadCifar10.load_cifar10()




