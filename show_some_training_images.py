import matplotlib.pyplot as plt
import numpy as np
from main import trainloader, classes, batch_size
from torchvision.utils import make_grid


# function to show images
def imgshow(img):
    img = 0.5 * img + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

imgshow(make_grid(images))

print(''.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))