import torch
import torch.nn as nn
import torch.optim as optim

import wandb
wandb.init()

from EEGConvNet import EEGConvNet
from dataset import create_data_loader

# Hyperparameters
lr = 0.0001
wd = 0.0001 # weight decay
num_epochs = 10
l = 1001 # length
d = 32 # reduced dimension
M = 24 # number of canonical channels
N = 22 # number of original channels
b = 64 # batch size

# get dataset
train_dataloader = create_data_loader(b)

# no mapping
model = EEGConvNet(use_mapping=False, dimension=l, length=l, num_channels=N, num_classes=3)
# with charm mapping
# model = EEGConvNet(use_mapping=True, length=l, original_channels=N, dimension=d, num_channels=M, num_classes=3)


print(model)
# nn.init.xavier_uniform_(model.embedding_layer.weight)  
# nn.init.xavier_uniform_(model.map_layer.weight)  

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


# Training loop
# torch.autograd.set_detect_anomaly(True):
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.float()
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        # Log loss and accuracy to wandb
        accuracy = (output.argmax(dim=1) == labels).float().mean()
        wandb.log({"Loss": loss.item(), "Accuracy": accuracy.item()})

        print(f"Epoch: {epoch+1}, batch: {i+1}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")

wandb.finish()
