import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

class ShowImages:

#   function to show images
    def imgshow(self, img):
        img = 0.5 * img + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_some_training_images(self):
        from loading_cifar10 import LoadCifar10
        batch_size, classes, trainloader, _ = LoadCifar10().load_cifar10()
        dataiter = iter(trainloader)
        images, labels = next(dataiter)

        self.imgshow(make_grid(images))

        print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))




