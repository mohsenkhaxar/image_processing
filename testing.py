import torch
from torchvision.utils import make_grid
from image_processing.loading_cifar10 import testloader, classes, batch_size
from image_processing.show_some_training_images import imshow
from image_processing.convolutional_neural_network import Net
from image_processing.training import PATH

# load trained model
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
print('model loaded')

# show images from test set
dataiter = iter(testloader)
images, labels = next(dataiter)
imshow(make_grid(images))
print('ground truth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# show predicted images on test set
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))