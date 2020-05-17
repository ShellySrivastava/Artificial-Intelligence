import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F

import matplotlib.pyplot as plt

train_set = torchvision.datasets.FashionMNIST(root = ".", train = True ,
download = True , transform = transforms.ToTensor())
test_set = torchvision.datasets.FashionMNIST(root = ".", train = False ,
download = True , transform = transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set , batch_size = 32,
shuffle = False)
test_loader = torch.utils.data.DataLoader(test_set , batch_size = 32,
shuffle = False)
torch.manual_seed(0)

# set true if using on google colab with runtime type = GPU
use_cuda = True

# defining neural network and the learnable parameters
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 32 output channels with a 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        # xavier initialisation for weights
        nn.init.xavier_normal_(self.conv1.weight)
        # 32 input image channel, 64 output channels with a 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 5)
        nn.init.xavier_normal_(self.conv2.weight)
        # an affine operation: y = Wx + b
        # fully connected layer 1
        self.fc1 = nn.Linear(64 * 4 * 4, 256) 
        nn.init.xavier_normal_(self.fc1.weight)
        # fully connected layer 2
        self.fc2 = nn.Linear(256, 10)
        nn.init.xavier_normal_(self.fc2.weight)
        # dropout layer
        # self.dropout = nn.Dropout(0.3) 

    def forward(self, x):
        # using relu activation after convolution and then max pooling over a 2x2 window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # flatten the tensors
        x = x.view(-1, self.num_flat_features(x))
        # using relu activation after fully connected layer
        x = F.relu(self.fc1(x))
        # output of the last layer - doesn't require softmax activation
        x = self.fc2(x)
        # dropout on the 2nd fully connected layer
        # x = self.dropout(self.fc2(x))
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


model = Net()

# use google colab gpu if resources are available
if use_cuda and torch.cuda.is_available():
    model.cuda()

# defining learning rate
learning_rate = 0.1
# defining loss function
criterion = nn.CrossEntropyLoss()
# defining stochastic gradient descent for weights update
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# evaluation method to calculate the accuracy of the model on training and test set
def evaluation(model, data_loader):
  # change the mode of the model to evaluation
  model.eval()
  total, correct = 0, 0
  for data in data_loader:
    inputs, label = data
    # use google colab gpu if resources are available
    if use_cuda and torch.cuda.is_available():
            inputs = inputs.cuda()
            label = label.cuda()
    outputs = model(inputs)
    _, pred = torch.max(outputs.data, 1)
    total = total + label.size(0)
    correct = correct + (pred == label).sum().item()
  # return accuracy
  return 100* (correct/total)

# total loss list for epochs
total_loss_list = []
# training accuracy list
train_acc_list = []
# testing accuracy list
test_acc_list = []
# epochs
num_epochs = 50

# training loop
for epoch in range(num_epochs):
    # loss list for batches
    loss_list = []
    for i, (images, labels) in enumerate(training_loader):
        # change the mode of the model to training
        model.train()
        # use google colab gpu if resources are available
        if use_cuda and torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        # Run the forward pass
        outputs = model(images)
        # calculate loss
        loss = criterion(outputs, labels)
        # append loss to the loss list
        loss_list.append(loss.item())

        # set gradient buffers to zero
        optimizer.zero_grad()
        # Backprop and perform SGD optimisation
        loss.backward()
        optimizer.step()

    # get training accuracy
    train_acc = evaluation(model, training_loader)
    # get testing accuracy
    test_acc = evaluation(model, test_loader)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    # get loss for the epoch and append to the total loss list
    total_loss_list.append(sum(loss_list))
    print('Epoch [{}/{}], Loss: {:.4f}, Train Accuracy: {:.2f}%, Test Accuracy: {:.2f}%'
          .format(epoch + 1, num_epochs, sum(loss_list), train_acc, test_acc))

plt.plot(train_acc_list, label="Train Acc")
plt.plot(test_acc_list, label="Test Acc")
plt.title('Test and Train Accuracy at LR={}'.format(learning_rate))
plt.legend()
plt.show()

plt.plot(total_loss_list, label="Loss")
plt.title('Loss per epoch at LR={}'.format(learning_rate))
plt.legend()
plt.show()
