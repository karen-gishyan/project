import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Data


class Network(nn.Module):
    """
    The network should have three subnetworks.
    Each subnetwork should minimize the loss between timesteps (i and i+1).
    Each loss of each subnetwork except the final one should minimize two losses (
        action_loss (drugs) and feature_loss(heart_rate). The final subnetwork should
        minimize the loss between the target hospial outcome).
    The final network should be a combination of the three networks, should output
    drugs, features for t1, and t2 and final discharge location for t3.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction
    def forward(self, x):
        yhat = torch.relu(self.linear(x))
        return yhat

def train():
    """
    Subnetwork 1 (t1) and subnetwork2 (t2) each train two models, and the final trains trains
    1 model.
    """
    def train_subnetwork(model, criterion, train_loader1,optimizer,train_loader2=None, epochs=10):
        """
        Each subnetwork minimizes cobmined loss of two models.
        """
        total_loss = []
        epoch_loss=[]
        ACC = []
        if train_loader2:
            for epoch in range(epochs):
                for i,((x1,y1), (x2,y2)) in enumerate(zip(train_loader1,train_loader2)):
                    yhat1 = model(x1)
                    yhat2 = model(x2)
                    loss = criterion(yhat1, y1)+criterion(yhat2, y2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss.append(loss.item())
                epoch_loss.append(loss.item())
                print(f'Epoch {epoch+1} completed.')
        else:
            # if the final subnetwork
            for epoch in range(epochs):
                for i,(x,y) in enumerate((train_loader1)):
                    yhat = model(x)
                    loss = criterion(yhat, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss.append(loss.item())
                epoch_loss.append(loss.item())
                print(f'Epoch {epoch+1} completed.')

        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot(total_loss, color=color)
        ax1.set_xlabel('Iteration', color=color)
        ax1.set_ylabel('total loss', color=color)
        ax1.tick_params(axis='y', color=color)

        plt.show()
        return total_loss

    # number of feature columns is 6(input dim), number of output columns is 3(output dim).
    model1 = Network(3,3)
    model2 = Network(6,3)
    final_model=Network(6,1)

    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.001)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.001)
    optimizer3 = torch.optim.SGD(final_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset1_1 = Data(timestep='t1',return_feature=True)
    dataset1_2 = Data(timestep='t1',return_feature=False)
    dataset2_1 = Data(timestep='t2',return_feature=True)
    dataset2_2 = Data(timestep='t2',return_feature=False)
    dataset_final=Data(timestep='t3')

    train_loader1_1 = DataLoader(dataset=dataset1_1, batch_size=10)
    train_loader1_2 = DataLoader(dataset=dataset1_2, batch_size=10)
    train_loader2_1 = DataLoader(dataset=dataset2_1, batch_size=10)
    train_loader2_2 = DataLoader(dataset=dataset2_2, batch_size=10)
    train_loader_final = DataLoader(dataset=dataset_final, batch_size=10)

    #train 1st subnetwork
    train_subnetwork(model1, criterion, train_loader1_1, optimizer1,train_loader2=train_loader1_2, epochs=10)
    #train 2nd subnetwork
    train_subnetwork(model2, criterion, train_loader2_1,optimizer2,train_loader2=train_loader2_2,epochs=10)
    #train 3rd subnetwork
    train_subnetwork(final_model, criterion, train_loader_final, optimizer3, epochs=10)

train()
