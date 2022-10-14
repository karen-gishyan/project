from helpers import configure_logger, create_split_loaders, accuracy, pred_to_labels, \
    count_uniques_in_pred_and_output
import matplotlib.pyplot as plt
from statistics import mean
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Data, SplitRNNData
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.nn.functional import softmax,sigmoid

logger=configure_logger()
torch.seed()

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
        x =self.classifier(torch.cat((x3, x4), dim=1))
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
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_first=True,
                 apply_softmax=True):
        super().__init__()
        self.rnn=nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=batch_first)
        self.linear=nn.Linear(hidden_size,output_size)
        self.apply_softmax=apply_softmax

    def forward(self,X,hidden_state):
        # r_out (batch, time_step, hidden_size)

        r_out,h_state=self.rnn(X,hidden_state)
        outs=[]

        for time_step in range(r_out.size(1)):
            if self.apply_softmax:
                out=torch.softmax(self.linear(r_out[:, time_step, :]),dim=-1)
            else:
                out=self.linear(r_out[:, time_step, :])
            outs.append(out)
        return torch.stack(outs, dim=1), h_state

class RNNEnsemble(nn.Module):
    def __init__(self,model_feature,model_drug,model_otput):
        super().__init__()
        self.model_feature=model_feature
        self.model_drug=model_drug
        self.model_otput=model_otput

    def forward(self,X1_t1,X2_t1,X1_t2,X2_t2,X1_t3,X2_t3,hidden_state):
        """
        Use timestep t features and drugs to predict t+1 features,drugs and output.
        Args:
            X1_t1 (torch.Tensor): Timestep-1 features
            X2_t1 (torch.Tensor): Timestep-1 drugs
            X1_t2 (torch.Tensor): Timestep-2 features
            X2_t2 (torch.Tensor): Timestep-2 drugs
            X1_t3 (torch.Tensor): Timestep-3 features
            X2_t3 (torch.Tensor): Timestep-3 drugs
            hidden_state (torch.Tensor): Hidden state
        Returns:
            Predicted features, drugs, output and hidden state
        """

        X1_t1_pred,hidden_state=self.model_feature(torch.cat((X1_t1,X2_t1),dim=2),hidden_state)
        X2_t1_pred,hidden_state=self.model_drug(torch.cat((X1_t1,X2_t1),dim=2),hidden_state)
        X1_t2_pred,hidden_state=self.model_feature(torch.cat((X1_t2,X2_t2),dim=2),hidden_state)
        X2_t2_pred,hidden_state=self.model_drug(torch.cat((X1_t2,X2_t2),dim=2),hidden_state)
        output,hidden_state=self.model_otput(torch.cat((X1_t3,X2_t3),dim=2),hidden_state)

        return X1_t1_pred,X2_t1_pred,X1_t2_pred,X2_t2_pred,output,hidden_state

    # untested
    def evaluation_forward(self,X1,X2,hidden_state, timestep:int):
        """
        Supply drugs, features and timestep for predicting next timestep's drugs, features
        or output. Subset method of forward().
        Args:
            X1 (torch.Tensor): Features
            X2 (torch.Tensor): Drugs
            timestep (int): timestep for making prediction.

        Returns:
            [torch.Tensor,torch.Tensor,Optional[torch.Tensor]]: Return output, hidden_state or
            X1_pred,X2_pred and hidden state, depending on the timestep.
        """
        if timestep==3:
            output,hidden_state=self.model_otput(torch.cat((X1,X2),dim=2),hidden_state)
            return output, hidden_state
        else:
            X1_pred,hidden_state=self.model_feature(torch.cat((X1,X2),dim=2),hidden_state)
            X2_pred,hidden_state=self.model_drug(torch.cat((X1,X2),dim=2),hidden_state)
            return X1_pred,X2_pred, hidden_state


    def sequential_evaluation_forward(self,X1_t1,X2_t1,hidden_state):
        """
        Perform successive forward using only timestep 1 features.
        This will be used for evaluating trained subnetworks only on the output.
        Here you rely only on your input features to do drug feature prediction for
        all successive timesteps, including the final output.
        """
        X1_t1_pred,hidden_state=self.model_feature(torch.cat((X1_t1,X2_t1),dim=2),hidden_state)
        X2_t1_pred,hidden_state=self.model_drug(torch.cat((X1_t1,X2_t1),dim=2),hidden_state)

        X1_t2_pred,hidden_state=self.model_feature(torch.cat((X1_t1_pred,X2_t1_pred),dim=2),hidden_state)
        X2_t2_pred,hidden_state=self.model_drug(torch.cat((X1_t1_pred,X2_t1_pred),dim=2),hidden_state)
        output,hidden_state=self.model_otput(torch.cat((X1_t2_pred,X2_t2_pred),dim=2),hidden_state)

        return output,hidden_state

#input of each network is dim(features)+dim(drugs)
rnn_feature=RNNNetwork(input_size=912,hidden_size=32,num_layers=1,output_size=10,apply_softmax=False)
rnn_drug=RNNNetwork(input_size=912,hidden_size=32,num_layers=1,output_size=902)
rnn_output=RNNNetwork(input_size=912,hidden_size=32,num_layers=1,output_size=13)

ensemble_model=RNNEnsemble(rnn_feature,rnn_drug,rnn_output)
optimizer = torch.optim.Adam(ensemble_model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(list(rnn_feature.parameters())+list(rnn_drug.parameters())+list(rnn_output.parameters()), lr=0.0001)
regression_loss=classification_loss= nn.MSELoss()
# #t1
# X1_t1=RNNData(is_feature=True,timestep=1)
# X2_t1=RNNData(is_feature=False,timestep=1)
# #t2
# X1_t2=RNNData(is_feature=True,timestep=2)
# X2_t2=RNNData(is_feature=False,timestep=2)
# #t3
# X1_t3=RNNData(is_feature=True,timestep=3)
# X2_t3=RNNData(is_feature=False,timestep=3)

# #t1 dataloader
# X1_t1_loader=DataLoader(dataset=X1_t1, batch_size=10)
# X2_t1_loader=DataLoader(dataset=X2_t1, batch_size=10)
# #t2 dataloader
# X1_t2_loader=DataLoader(dataset=X1_t2, batch_size=10)
# X2_t2_loader=DataLoader(dataset=X2_t2, batch_size=10)
# #t3 dataloader
# X1_t3_loader=DataLoader(dataset=X1_t3, batch_size=10)
# X2_t3_loader=DataLoader(dataset=X2_t3, batch_size=10)

#when selecting the batch size, make sure the data length is divisible by batch size.
X1_t1_train,X1_t1_test,X1_t1_valid=create_split_loaders(is_feature=True,timestep=1,batch_size=159)
X2_t1_train,X2_t1_test,X2_t1_valid=create_split_loaders(is_feature=False,timestep=1,batch_size=159)
X1_t2_train,X1_t2_test,X1_t2_valid=create_split_loaders(is_feature=True,timestep=2,batch_size=159)
X2_t2_train,X2_t2_test,X2_t2_valid=create_split_loaders(is_feature=False,timestep=2,batch_size=159)
X1_t3_train,X1_t3_test,X1_t3_valid=create_split_loaders(is_feature=True,timestep=3,batch_size=159)
X2_t3_train,X2_t3_test,X2_t3_valid=create_split_loaders(is_feature=False,timestep=3,batch_size=159)


def train_rnn_ensemble(epochs=100):
    total_loss=[]
    l11_epoch_loss,l21_epoch_loss,lout_epoch_loss=[],[],[]
    l12_epoch_loss,l22_epoch_loss=[],[]

    h_state = None
    # any other load batch size could have been taken
    batch_size=159
    for epoch in range(epochs):
        l11_batch_loss,l21_batch_loss,lout_batch_loss=[],[],[]
        l12_batch_loss,l22_batch_loss=[],[]
        for i,((x1_t1,y11), (x2_t1,y21),(x1_t2,y12),(x2_t2,y22), (x1_t3,y),(x2_t3,y)) \
                 in enumerate(zip(X1_t1_train,X2_t1_train,X1_t2_train,\
                     X2_t2_train,X1_t3_train,X2_t3_train)):

            # results of subnetworks training
            X1_t1_pred,X2_t1_pred, X1_t2_pred, X2_t2_pred, output,hidden_state=\
                ensemble_model(x1_t1,x2_t1,x1_t2,x2_t2,x1_t3,x2_t3,h_state)

            # type comes first (drug/feature), timestep comes second in this l logic
            #feature_t1
            l11=regression_loss(X1_t1_pred,y11)
            l11_batch_loss.append(l11.item())
            #feature_t2
            l12=regression_loss(X1_t2_pred,y12)
            l12_batch_loss.append(l12.item())

            #drug_t1
            l21=classification_loss(X2_t1_pred,y21)
            l21_batch_loss.append(l21.item())
            #drug_t2
            l22=classification_loss(X2_t2_pred,y22)
            l22_batch_loss.append(l22.item())
            #output
            lout=classification_loss(output,y)
            lout_batch_loss.append(lout.item())

            # combined weighted loss of training subnetworks.
            # each loss updates only the weights of its relevant network(tested),
            # for that reason combining losses makes sense.
            #TODO some updates happen for other networks as well, needs to be investigated.
            loss=(l11+l12+l21+l22+lout)/batch_size

            # for lstm
            # h_state = hidden_state[0].data
            h_state=list(hidden_state)
            h_state[0]=hidden_state[0].data
            h_state[1]=hidden_state[1].data

            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(ensemble_model.parameters(), 0.001)
            #TODO step of one model slighly affects other models as well.
            optimizer.step()
            total_loss.append(loss.item())
            # logger.info(f'Epoch {epoch+1},Step {i+1}, Loss-Value {l11.item()}')
        print(f"Epoch {epoch+1} completed.")

        if l11_batch_loss: l11_epoch_loss.append(mean(l11_batch_loss))
        if l21_batch_loss: l21_epoch_loss.append(mean(l21_batch_loss))
        if l12_batch_loss: l12_epoch_loss.append(mean(l12_batch_loss))
        if l22_batch_loss: l22_epoch_loss.append(mean(l22_batch_loss))
        if lout_batch_loss: lout_epoch_loss.append(mean(lout_batch_loss))
        # print(f'Epoch: {epoch+1}, Loss Value:{loss.item()}')
        # logger.info(f'Epoch: {epoch+1}, Drug t1 Average Batch Loss Value:{l12}')
        # logger.info(f'Epoch: {epoch+1}, Drug t2 Average Batch Loss Value:{l22}')

    _, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5, sharey=False)
    ax1.plot(l11_epoch_loss)
    ax1.set_title('t1 Features')
    ax2.plot(l12_epoch_loss)
    ax2.set_title('t1 Drugs')
    ax3.plot(l21_epoch_loss)
    ax3.set_title('t2 Features')
    ax4.plot(l22_epoch_loss)
    ax4.set_title('t2 Drugs')
    ax5.plot(lout_epoch_loss)
    ax5.set_title('Discharge')
    plt.show()

    return hidden_state

logger.info("Accuracy Before training.")
### Validation (Regular Forward)
t1_feature=SplitRNNData(is_feature=True,timestep=1,split='valid')
t1_drug=SplitRNNData(is_feature=False,timestep=1,split='valid')
t2_feature=SplitRNNData(is_feature=True,timestep=2,split='valid')
t2_drug=SplitRNNData(is_feature=False,timestep=2,split='valid')
t3_feature=SplitRNNData(is_feature=True,timestep=3,split='valid')
t3_drug=SplitRNNData(is_feature=False,timestep=3,split='valid')

hidden_state=None
# forward
X1_t1_pred,X2_t1_pred,X1_t2_pred,X2_t2_pred,output,hidden_state=\
    ensemble_model(t1_feature.X1_feature,t1_drug.X1_drug,t2_feature.X2_feature,t2_drug.X2_drug,t3_feature.X3_feature,t3_drug.X3_drug,hidden_state)

logger.info("Feature_t1:{}".format(accuracy(X1_t1_pred,t1_feature.Y1_feature,feature=True)))
logger.info("Drug_t1:{}".format(accuracy(X2_t1_pred,t1_drug.Y1_drug)))
logger.info("Feature_t2:{}".format(accuracy(X1_t2_pred,t2_feature.Y2_feature,feature=True)))
logger.info("Drug_t2:{}".format(accuracy(X2_t2_pred,t2_drug.Y2_drug)))
# last two should provide the same results.
logger.info('Output:{}'.format(accuracy(output,t3_feature.Y3_feature)))
# print(accuracy(output,t3_drug.Y3_drug))

# pred_counts, output_counts=count_uniques_in_pred_and_output(output,t3_feature.Y3_feature)

#train
hidden_state=train_rnn_ensemble()

logger.info('After Training.')
X1_t1_pred,X2_t1_pred,X1_t2_pred,X2_t2_pred,output,hidden_state=\
    ensemble_model(t1_feature.X1_feature,t1_drug.X1_drug,t2_feature.X2_feature,t2_drug.X2_drug,t3_feature.X3_feature,t3_drug.X3_drug,hidden_state)

#TODO drugs are not learning well, loss curves should be improved.
logger.info("Feature_t1:{}".format(accuracy(X1_t1_pred,t1_feature.Y1_feature,feature=True)))
logger.info("Drug_t1:{}".format(accuracy(X2_t1_pred,t1_drug.Y1_drug)))
logger.info("Feature_t2:{}".format(accuracy(X1_t2_pred,t2_feature.Y2_feature,feature=True)))
logger.info("Drug_t2:{}".format(accuracy(X2_t2_pred,t2_drug.Y2_drug)))
# last two should provide the same results.
logger.info('Output:{}'.format(accuracy(output,t3_feature.Y3_feature)))
