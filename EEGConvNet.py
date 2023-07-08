import torch
import torch.nn as nn

class EEGConvNet(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(EEGConvNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.instance_norm = nn.InstanceNorm1d(num_channels, affine=True)
        self.conv_block1 = self._create_conv_block(num_channels, features=256, kernel=8, conv_stride=1, pooling=2, pool_stride=2)
        self.conv_block2 = self._create_conv_block(256, features=256, kernel=3, conv_stride=1, pooling=2, pool_stride=2)
        self.conv_block3 = self._create_conv_block(256, features=256, kernel=3, conv_stride=1, pooling=2, pool_stride=2)
        self.conv_block4 = self._create_conv_block(256, features=512, kernel=3, conv_stride=1, pooling=2, pool_stride=2)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        

    def _create_conv_block(self, in_channels, features, kernel, conv_stride, pooling, pool_stride):
        return nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=kernel, stride=conv_stride, padding=1),
            nn.GroupNorm(1, features),
            nn.PReLU(),
            nn.MaxPool1d(pooling, pool_stride)
        )

    def forward(self, inp):
        print(inp.shape)
        x = self.instance_norm(inp)
        print(x.shape)
        x = self.conv_block1(x)
        print(x.shape)
        x = self.conv_block2(x)
        print(x.shape)
        x = self.conv_block3(x)
        print(x.shape)
        x = self.conv_block4(x)
        print("...")
        print(x.shape)
        # x = self.max_pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        
        return x
    
d = 32 # reduced dimension
M = 24 # number of channels
b = 3 # batch size
model = EEGConvNet(num_channels=M, num_classes=3)
print(model)
# nn.init.xavier_uniform_(model.embedding_layer.weight)  
# nn.init.xavier_uniform_(model.map_layer.weight)  
x = torch.rand(b, M, d)
out = model(x)
print(out.shape)