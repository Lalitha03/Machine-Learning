#Importing libraries
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.autograd import Variable

#Initializing hyperparameters
num_epochs = 10
num_classes = 10
batch_size = 100
learning_rate = 0.001

#Loading dataset
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.FashionMNIST(root='./data', 
                            train=True, 
                            download=True,
                            transform=transform
                            )

test_dataset = datasets.FashionMNIST(root='./data', 
                           train=False, 
                           download=True,
                           transform=transform
                           )

#Loading dataset into dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

#Defining the network          
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        #Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        
        #Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        #Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        
        #Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        #Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)
        
        #Fully Connected 1
        self.fc1 = nn.Linear(32*7*7, 10)
    
    def forward(self, x):
        #Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        #Max pool 1
        out = self.maxpool1(out)
        
        #Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
        
        #Max pool 2
        out = self.maxpool2(out)
        
        #Resize
        out = out.view(out.size(0), -1)
        
        #Dropout
        out = self.dropout(out)
        
        #Fully connected 1
        out = self.fc1(out)
        
        return out

#Create instance of model
model = CNNModel()

#Create instance of loss
criterion = nn.CrossEntropyLoss()

#Create instance of optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#Train the model
iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        #Clear the gradients
        optimizer.zero_grad()
        
        #Forward propagation 
        outputs = model(images)      
        
        #Calculating loss with softmax to obtain cross entropy loss
        loss = criterion(outputs, labels)
        
        #Backward propation
        loss.backward()
        
        #Updating gradients
        optimizer.step()
        
        iter += 1
        
        #Total number of labels
        total = labels.size(0)
        
        #Obtaining predictions from max value
        _, predicted = torch.max(outputs.data, 1)
        
        #Calculate the number of correct answers
        correct = (predicted == labels).sum().item()
        
        #Print loss and accuracy
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(),
                          (correct / total) * 100))

#Testing the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
