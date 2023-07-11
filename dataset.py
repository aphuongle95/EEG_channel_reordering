import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import moabb
from moabb.datasets import BNCI2014001
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery



class CustomDataset(Dataset):
    def __init__(self, inputs, labels):
        self.data = torch.tensor(inputs)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        label = self.labels[index]
        return data_sample, label

def create_data_loader(batch_size):
    paradigm = LeftRightImagery()
    print(paradigm.datasets)
    # Get dataset
    dataset = BNCI2014001()
    dataset.subject_list = [1] # quick 
    X, labels, meta = paradigm.get_data(dataset=dataset)
    # X = X[:, :, :100] # quick, rm when data is mapped

    all_labels = {'right_hand': 0, 'left_hand': 1}
    labels= [all_labels[i] for i in labels]
    dataset = CustomDataset(X, labels)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader
    
if __name__ == "__main__":
    paradigm = LeftRightImagery()
    print(paradigm.datasets)
    # Get dataset
    dataset = BNCI2014001()
    dataset.subject_list = [1] # quick 
    X, labels, meta = paradigm.get_data(dataset=dataset)

    all_labels = {'right_hand': 0, 'left_hand': 1}
    labels= [all_labels[i] for i in labels]
    dataset = CustomDataset(X, labels)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.shape)
    print(train_labels.shape)