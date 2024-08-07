import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# Load the dataset
dat_dir = r'C:\Users\naras\.spyder-py3\document'
trainset = datasets.ImageFolder(dat_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testset = datasets.ImageFolder(dat_dir, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

# Ensure we only have 3 classes
assert len(trainset.classes) == 3, f"Expected 3 classes, but found {len(trainset.classes)}"

# Visualize some images
dataiter = iter(trainloader)
images, labels = next(dataiter)  # Use next() directly

fig, axes = plt.subplots(figsize=(12, 6), nrows=2, ncols=4)
for ii, (image, label) in enumerate(zip(images[:8], labels[:8])):
    ax = axes[ii // 4, ii % 4]
    ax.imshow(image.permute(1, 2, 0))
    ax.set_title(f'Label: {trainset.classes[label]}')
    ax.axis('off')
plt.show()

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1)  # Change input channels to 3
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=2, padding=1)
        self.conv2_drop = nn.Dropout2d()  # Regularization
        
        # Calculate the size of the input to the first fully connected layer
        self._to_linear = None
        self.convs(torch.randn(1, 3, 224, 224))
        
        self.fc1 = nn.Linear(self._to_linear, 50)  # Adjust input size to fully connected layer
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30, 3)  # Change output size to 3 for 3 classes

    def convs(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv3(x)), 2))
        if self._to_linear is None:
            self._to_linear = x.view(-1).shape[0]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # Flatten tensor
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

net = Net()
print(net)

import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.NLLLoss()

epochs = 30
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    net.train()
    for images, labels in trainloader:
        # Clear the gradients
        optimizer.zero_grad()
        # Forward pass, get our logits
        log_ps = net(images)
        # Calculate the loss with the logits and the labels
        loss = criterion(log_ps, labels)
        # Calculate the gradients
        loss.backward()
        # Update the weights
        optimizer.step()
        
        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        net.eval()
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = net(images)
                test_loss += criterion(log_ps, labels).item()
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))
        # print training/test statistics
        print(f"Epoch: {e+1}/{epochs}.. "
              f"Training Loss: {running_loss/len(trainloader):.3f}.. "
              f"Test Loss: {test_loss/len(testloader):.3f}.. "
              f"Test Accuracy: {accuracy/len(testloader):.3f}")

# Plotting training and test losses
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.legend(frameon=False)
plt.show()


examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
    output = net(example_data)
    
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])