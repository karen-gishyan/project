import os
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

    def train_subnetwork(model1, criterion, train_loader1,optimizer,model2=None,train_loader2=None, epochs=10):
        """
        Each subnetwork minimizes cobmined loss of two models.
        """

        total_loss = []
        epoch_loss=[]
        ACC = []
        if train_loader2:
            for epoch in range(epochs):
                for i,((x1,y1), (x2,y2)) in enumerate(zip(train_loader1,train_loader2)):
                    yhat1 = model1(x1)
                    yhat2 = model2(x1)
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
                    yhat = model1(x)
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

    #train 1st subnetwork
    train_subnetwork(model1,criterion, train_loader1_1, optimizer1,model2=model2,train_loader2=train_loader1_2, epochs=10)
    #train 2nd subnetwork
    train_subnetwork(model3, criterion, train_loader2_1,optimizer2,model2=model4,train_loader2=train_loader2_2,epochs=10)
    #train 3rd subnetwork
    train_subnetwork(final_model, criterion, train_loader_final, optimizer3, epochs=10)

    return model1, model2, model3, model4, final_model

# number of feature columns is 6(input dim), number of output columns is 3(output dim).
model1=model2= Network(3,3)
model3=model4= Network(6,3)
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

# model1, model2, model3, model4, final_model=train()
# # prediction
# with torch.no_grad():
#     features_t0,_=next(iter(train_loader1_1))
#     features_t1=model1(features_t0)
#     drugs_t1=model2(features_t0)
#     features_t2=model3(torch.concat([features_t1,drugs_t1],dim=1))
#     drugs_t2=model4(torch.concat([features_t1,drugs_t1],dim=1))
#     final_outcome=final_model(torch.concat([features_t2,drugs_t2],dim=1))

###### Approach 2 #######
class Ensemble(nn.Module):
    def __init__(self, model1, model2,model3,model4):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.classifier = nn.Linear(6, 1)

    def forward(self, x1, x2):
        """
        Parameters:
            x1: train_loader1 from timestep1.
            x2: train_loader2 from timestep2.
        The final outcome is trained not on the t3 actual results,
        but on the predicted results of the previous timesteps.
        t3 pred obtained from t2 models, t2 pred obtained from t1 models.
        """
        #t1
        x1 = self.model1(x1)
        x2 = self.model2(x2)
        #t2
        x3 = self.model3(torch.cat((x1, x2), dim=1))
        x4 = self.model4(torch.cat((x1, x2), dim=1))
        #t3
        x =self.classifier(torch.relu(torch.cat((x3, x4), dim=1)))
        return x


# model1, model2, model3, model4, _=train()

dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'weights')
os.chdir(dir_)

# torch.save(model1.state_dict(), 'model1.ckpt')
# torch.save(model2.state_dict(), 'model2.ckpt')
# torch.save(model3.state_dict(), 'model3.ckpt')
# torch.save(model4.state_dict(), 'model4.ckpt')

def train_ensemble(model,train_loader1,train_loader2,optimizer,epochs=10):
    total_loss = []
    epoch_loss=[]
    for epoch in range(epochs):
        for i,((x11,_), (x12,_),(_,y3)) in enumerate(zip(train_loader1,train_loader2,train_loader_final)):
            yhat = model(x11,x12)
            loss = criterion(yhat,y3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        epoch_loss.append(loss.item())
        print(f'Epoch {epoch+1} completed.')

    _, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(total_loss, color=color)
    ax1.set_xlabel('Iteration', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)
    plt.show()

model1.load_state_dict(torch.load('model1.ckpt'))
model2.load_state_dict(torch.load('model2.ckpt'))
model3.load_state_dict(torch.load('model3.ckpt'))
model4.load_state_dict(torch.load('model4.ckpt'))

ensemble_model=Ensemble(model1,model2,model3,model4)
optimizer=torch.optim.SGD(ensemble_model.parameters(), lr=0.001)
train_ensemble(ensemble_model,train_loader1_1,train_loader1_2,optimizer)