import torch
from image_processing.loading_cifar10 import testloader
from image_processing.convolutional_neural_network import Net
from image_processing.training import PATH


# load trained model
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
print('model loaded')

# calculate net accuracy on whole test data
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
