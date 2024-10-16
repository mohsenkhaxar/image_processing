import torch
from image_processing.loading_cifar10 import classes, testloader
from image_processing.convolutional_neural_network import Net
from image_processing.training import PATH


# load trained model
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
print('model loaded')

# calculate performance
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname:0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'accuracy for class: {classname:5s} is {accuracy:.1f} %')
