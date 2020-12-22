from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class IrisDataSet(Dataset):
    def __init__(self, X, y=None, transform=None):
        self.X = torch.tensor(X).float()
        self.y = y
        if y is not None:
            self.y = torch.tensor(y).long()
        self.transform = transform

    def __len__(self):
        return len(self.X[:, 0])

    def __getitem__(self, index):
        image = self.X[index, :]

        if self.transform is not None:
            image = self.transform(image)

        if self.y is not None:
            return image, self.y[index]
        else:
            return image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(),
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


if __name__ == "__main__":
    # Load iris dataset
    data = load_iris()

    # Extract features, targets and descriptions
    FeatureNames = data.feature_names
    X = np.array(pd.DataFrame(data.data))
    # y = np.array(pd.get_dummies(pd.DataFrame(data.target).iloc[:, 0]))
    y = np.array(pd.DataFrame(data.target).iloc[:, 0])
    TargetNames = data.target_names

    # Split into train test samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # prepare the pytorch train / test classes
    traindataset = IrisDataSet(X=X_train, y=y_train, transform=None)
    trainloader = DataLoader(dataset=traindataset, batch_size=10, shuffle=True)

    # Initialise the network
    net = Net()

    # Specify a loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    print("Finished Training")

    # Now predict the test data (various ways to do it)
    # simple
    # test_outputs = net(torch.tensor(X_test).float())

    # complex but uses the dataloader classes so might be easier if lots of transform steps
    testdataset = IrisDataSet(X=X_test, y=None, transform=None)
    testloader = DataLoader(
        dataset=testdataset, batch_size=len(testdataset), shuffle=False
    )
    test_outputs = np.concatenate(
        [net(data).detach().cpu().numpy() for data in testloader], axis=0
    )
