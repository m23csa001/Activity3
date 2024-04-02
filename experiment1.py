#Write code here from class assignment 2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from torch.utils.tensorboard import SummaryWriter
# Data loading and preprocessing
transform = transforms.Compose([
   transforms.Resize((224, 224)),  # Resize images to fit ResNet101 input size
   transforms.ToTensor(),
   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


train_dataset = datasets.SVHN(root='data/', split='train', transform=transform, download=True)
test_dataset = datasets.SVHN(root='data/', split='test', transform=transform, download=True)


train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

#Model loading and modification
model = models.resnet50(pretrained=True)


# Freeze all layers in the network
for param in model.parameters():
   param.requires_grad = False


# Unfreeze the fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
for param in model.fc.parameters():
   param.requires_grad = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


criterion = nn.CrossEntropyLoss()
# EDA: Visualize some images from the SVHN dataset
def visualize_dataset(dataset, num_images=5):
   fig, ax = plt.subplots(1, num_images, figsize=(15, 3))
   for i in range(num_images):
       ax[i].imshow(np.transpose(dataset[i][0].numpy(), (1, 2, 0)))
       ax[i].set_title(f"Label: {dataset[i][1]}")
       ax[i].axis('off')
   plt.show()
visualize_dataset(train_dataset)
def train(model, train_loader, criterion, optimizer, num_epochs=10, optimizer_name='optimizer'):
   writer = SummaryWriter(f'runs/SVHN_{optimizer_name}')
   for epoch in range(num_epochs):
       model.train()
       running_loss = 0.0
       correct = 0
       total = 0
       for batch_idx, (images, labels) in enumerate(train_loader):
           images, labels = images.to(device), labels.to(device)


           optimizer.zero_grad()
           outputs = model(images)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()


           running_loss += loss.item()
           _, predicted = outputs.max(1)
           total += labels.size(0)
           correct += predicted.eq(labels).sum().item()


       epoch_loss = running_loss / len(train_loader)
       epoch_accuracy = 100. * correct / total
       print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')


       writer.add_scalar('Training Loss', epoch_loss, epoch)
       writer.add_scalar('Training Accuracy', epoch_accuracy, epoch)


   writer.close()
def top_k_accuracy(model, test_loader, k=5):
   model.eval()
   correct = 0
   total = 0
   with torch.no_grad():
       for images, labels in test_loader:
           images, labels = images.to(device), labels.to(device)
           outputs = model(images)
           _, pred = outputs.topk(k, 1, True, True)
           correct += pred.eq(labels.view(-1, 1).expand_as(pred)).sum().item()
           total += labels.size(0)
   return 100. * correct / total
optimizers = {
   'Adam': optim.Adam(model.fc.parameters()),  # Only optimize fc layer parameters
   'Adagrad': optim.Adagrad(model.fc.parameters()),
   'Adadelta': optim.Adadelta(model.fc.parameters()),
}
for name, optimizer in optimizers.items():
   print(f"Training with {name}")
   train(model, train_loader, criterion, optimizer, num_epochs=10, optimizer_name=name)
   top5_acc = top_k_accuracy(model, test_loader)
   print(f'Top-5 Test Accuracy with {name}: {top5_acc}%\n')
