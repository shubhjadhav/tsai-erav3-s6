from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        
        # Convolution layers with fewer filters
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0) 
        self.conv2 = nn.Conv2d(16, 32, 3, padding=0)  
        self.conv3 = nn.Conv2d(32, 10, 1, padding=0)
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=0)
        self.conv7 = nn.Conv2d(16, 16, 3, padding=1)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Global Average Pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Batch Normalization and Dropout for regularization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(16)
        self.bn6 = nn.BatchNorm2d(16)
        self.bn7 = nn.BatchNorm2d(16)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.1)

        # Fully connected layer (output 10 classes)
        # Global Average Pooling output shape: (batch_size, 64, 1, 1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Apply convolutions, activations, and batch normalization
        x = self.dropout(self.bn1(F.relu(self.conv1(x))))
        x = self.dropout(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.conv3(x))
        x = self.dropout(self.bn4(F.relu(self.conv4(x))))
        x = self.dropout(self.bn5(F.relu(self.conv5(x))))
        x = self.dropout(self.bn6(F.relu(self.conv6(x))))
        x = self.dropout(self.bn7(F.relu(self.conv7(x))))
        
        # Apply Global Average Pooling (GAP)
        x = self.gap(x)
        
        # Flatten the output from GAP
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        
        # Fully connected layer for classification
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / len(train_loader.dataset)

    print('Training: Average loss: {:.4f}, Accuracy: {}/{} ({:.01f}%)'.format(
        train_loss, 
        correct, 
        len(train_loader.dataset),
        train_acc)
    )

    return train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    print('Testing: Average loss: {:.4f}, Accuracy: {}/{} ({:.01f}%)\n'.format(
        test_loss, 
        correct, 
        len(test_loader.dataset),
        test_acc)
    )

    return test_acc

def model_train_test():

    use_cuda = torch.cuda.is_available()
    print(f"\nGPU available: {use_cuda}")

    device = torch.device("cuda" if use_cuda else "cpu")

    model = MNIST().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)

    print(f'\nModel Summary')
    summary(model, input_size=(1, 28, 28))

    torch.manual_seed(1)
    batch_size = 128

    # Define data transformations with augmentations
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std (for MNIST)
    ])

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,transform=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs
    )

    train_acc_lst = []
    test_acc_lst = []

    for epoch in range(1, 21):
        print("\nEpoch {}".format(epoch))
        train_acc = train(model, device, train_loader, optimizer, epoch)
        train_acc_lst.append(train_acc)
        test_acc = test(model, device, test_loader)
        test_acc_lst.append(test_acc)

    return {
        'train_accuracy': max(train_acc_lst),
        'test_accuracy': max(test_acc_lst)
    }
        

if __name__ == '__main__':
    model_train_test()