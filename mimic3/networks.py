import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Data,RNNData


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

    def train_subnetwork(model1, criterion, train_loader1,optimizer1,model2=None,train_loader2=None,optimizer2=None, epochs=10):
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
                    yhat2 = model2(x2)
                    loss = criterion(yhat1, y1)+criterion(yhat2, y2)
                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()
                    total_loss.append(loss.item())
                epoch_loss.append(loss.item())
                print(f'Epoch {epoch+1} completed.')
        else:
            # if the final subnetwork
            for epoch in range(epochs):
                for i,(x,y) in enumerate((train_loader1)):
                    yhat = model1(x)
                    loss = criterion(yhat, y)
                    optimizer1.zero_grad()
                    loss.backward()
                    optimizer1.step()
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

    #uses variables from global context
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

optimizer1 = torch.optim.SGD(list(model1.parameters())+list(model2.parameters()), lr=0.001)
optimizer2 = torch.optim.SGD(list(model3.parameters())+list(model4.parameters()), lr=0.001)
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
    """
    Drawback of this approach is that features and drugs at none of the timestep are
    used for the loss calculation. Only the features, drugs at t_1 are used for predicting
    the final output.
    """
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

# dir_=os.path.dirname(__file__)
# dir_=os.path.join(dir_,'weights')
# os.chdir(dir_)

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

# model1.load_state_dict(torch.load('model1.ckpt'))
# model2.load_state_dict(torch.load('model2.ckpt'))
# model3.load_state_dict(torch.load('model3.ckpt'))
# model4.load_state_dict(torch.load('model4.ckpt'))

# ensemble_model=Ensemble(model1,model2,model3,model4)
# optimizer=torch.optim.SGD(ensemble_model.parameters(), lr=0.001)
# train_ensemble(ensemble_model,train_loader1_1,train_loader1_2,optimizer)


###### Approach 3 ####
class RNNNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_first=True):
        super().__init__()
        self.rnn=nn.RNN(input_size=input_size,hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=batch_first)
        self.linear=nn.Linear(hidden_size,output_size)

    def forward(self,X,hidden_state):
        # r_out (batch, time_step, hidden_size)
        r_out,h_state=self.rnn(X,hidden_state)
        outs=[]

        for time_step in range(r_out.size(1)):
            outs.append(self.linear(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

class RNNEnsemble(nn.Module):
    def __init__(self,model_feature,model_drug,model_otput):
        super().__init__()
        self.model_feature=model_feature
        self.model_drug=model_drug
        self.model_otput=model_otput

    def forward(self,X1_t1,X2_t1,X1_t2,X2_t2,X1_t3,X2_t3,hidden_state):
        X1_t1_pred,hidden_state=self.model_feature(torch.cat((X1_t1,X2_t1),dim=2),hidden_state)
        X2_t1_pred,hidden_state=self.model_drug(torch.cat((X1_t1,X2_t1),dim=2),hidden_state)
        X1_t2_pred,hidden_state=self.model_feature(torch.cat((X1_t2,X2_t2),dim=2),hidden_state)
        X2_t2_pred,hidden_state=self.model_drug(torch.cat((X1_t2,X2_t2),dim=2),hidden_state)
        output,hidden_state=self.model_otput(torch.cat((X1_t3,X2_t3),dim=2),hidden_state)

        return X1_t1_pred,X2_t1_pred,X1_t2_pred,X2_t2_pred,output,hidden_state

# in the prediction phase we assume the input drugs are known
rnn_feature=RNNNetwork(input_size=11,hidden_size=32,num_layers=1,output_size=10)
rnn_drug=RNNNetwork(input_size=11,hidden_size=32,num_layers=1,output_size=1)
rnn_output=RNNNetwork(input_size=11,hidden_size=32,num_layers=1,output_size=1)

ensemble_model=RNNEnsemble(rnn_feature,rnn_drug,rnn_output)
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.01)
# see which one works
# optimizer = torch.optim.SGD(list(rnn_feature.parameters())+list(rnn_drug.parameters())+list(rnn_output.parameters()), lr=0.01)

loss_func = nn.MSELoss()

#t1
X1_t1=RNNData(is_feature=True,timestep=1)
X2_t1=RNNData(is_feature=False,timestep=1)
#t2
X1_t2=RNNData(is_feature=True,timestep=2)
X2_t2=RNNData(is_feature=False,timestep=2)
#t3
X1_t3=RNNData(is_feature=True,timestep=3)
X2_t3=RNNData(is_feature=False,timestep=3)

#t1 dataloader
X1_t1_loader=DataLoader(dataset=X1_t1, batch_size=10)
X2_t1_loader=DataLoader(dataset=X2_t1, batch_size=10)
#t2 dataloader
X1_t2_loader=DataLoader(dataset=X1_t2, batch_size=10)
X2_t2_loader=DataLoader(dataset=X2_t2, batch_size=10)
#t3 dataloader
X1_t3_loader=DataLoader(dataset=X1_t3, batch_size=10)
X2_t3_loader=DataLoader(dataset=X2_t3, batch_size=10)


def train_rnn_ensemble(epochs=30):
    total_loss = []
    epoch_loss=[]
    h_state = None
    for epoch in range(epochs):
        for _,((x1_t1,y11), (x2_t1,y21),(x1_t2,y12),
             (x2_t2,y22), (x1_t3,y),(x2_t3,y)) \
                 in enumerate(zip(X1_t1_loader,X2_t1_loader,X1_t2_loader,X2_t2_loader,X1_t3_loader,X2_t3_loader)):
            X1_t1_pred,X2_t1_pred, X1_t2_pred, X2_t2_pred, output,hidden_state=\
                ensemble_model(x1_t1,x2_t1,x1_t2,x2_t2,x1_t3,x2_t3,h_state)
            h_state = hidden_state.data

            #TODO loss on this dataset should be improved
            # loss=criterion(X1_t2_pred,y12)
            loss=criterion(X1_t1_pred,y11)+criterion(X1_t2_pred,y12)+\
                criterion(X2_t1_pred,y21)+\
                criterion(X2_t2_pred,y22)+criterion(output,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        epoch_loss.append(loss.item())
        print(f'Epoch {epoch+1} completed.')

    _, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(epoch_loss, color=color)
    ax1.set_xlabel('Epoch', color=color)
    ax1.set_ylabel('Total Loss', color=color)

    ax1.tick_params(axis='y', color=color)
    plt.title('Ensemble Model Results')
    plt.show()


train_rnn_ensemble()
