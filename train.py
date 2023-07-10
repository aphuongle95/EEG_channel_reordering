import torch
import torch.nn as nn
import torch.optim as optim

import wandb
wandb.init()

from EEGConvNet import EEGConvNet
from dataset import train_dataloader


# Hyperparameters
lr = 0.0001
wd = 0.0001 # weight decay
num_epochs = 10
d = 32 # reduced dimension
# d = 1001
M = 24 # number of channels
b = 64 # batch size

model = EEGConvNet(num_channels=M, num_classes=3)
print(model)
# nn.init.xavier_uniform_(model.embedding_layer.weight)  
# nn.init.xavier_uniform_(model.map_layer.weight)  
x = torch.rand(b, M, d)
out = model(x)
print(out.shape)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

# Create random input tensor and labels
input_data = torch.rand(b, M, d)
labels = torch.randint(0, 3, (b,))
print(labels.shape)

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_dataloader):

        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # Log loss and accuracy to wandb
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        wandb.log({"Loss": loss.item(), "Accuracy": accuracy.item()})

        print(f"Epoch: {epoch+1}, batch: {i+1}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")

wandb.finish()
