import os
import yaml
import json
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from torch import sigmoid
import matplotlib.pyplot as plt


dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)

def combine_timestep_data_rnn_and_dummy_format(diagnosis):
    """
    The data in a dummy format is the input to the EvaluationBaseModel.
    """
    #rnn 3d format
    t1_drugs=torch.load(f"{diagnosis}/t1/drugs.pt")
    t2_drugs=torch.load(f"{diagnosis}/t2/drugs.pt")
    t3_drugs=torch.load(f"{diagnosis}/t3/drugs.pt")
    assert t1_drugs.shape[0]==t2_drugs.shape[0]==t3_drugs.shape[0],"unequal number of admissions"
    #2d
    combined_drugs=torch.cat((t1_drugs,t2_drugs,t3_drugs),dim=1)
    torch.save(combined_drugs.view(t1_drugs.shape[0],1,-1),f"{diagnosis}/combined-drugs-rnn-format.pt")

    #dummy (0,1) format, where data is 2d
    with open("../json/drug_mapping.json") as file:
        drug_mappings=json.load(file)

    len_total_drugs=len(list(drug_mappings))
    dummy_format=torch.zeros(t1_drugs.shape[0],len_total_drugs)

    # for a given rows 1 will indicate that drug was given, else we leave as 0
    for i, row in enumerate(combined_drugs):
        for drug_index in row:
            if drug_index!=-1:
                dummy_format[i][drug_index]=1

    torch.save(dummy_format,f"{diagnosis}/combined-drugs-dummy-format.pt")



class EvaluationDataset(Dataset):
    def __init__(self,diagnosis,split_type='train',data_format='rnn'):
        super().__init__()
        self.split_type=split_type
        if data_format=='rnn':
            self.X=torch.load(f"{diagnosis}/combined-drugs-rnn-format.pt")
        elif data_format=='dummy':
            self.X=torch.load(f"{diagnosis}/combined-drugs-dummy-format.pt")
        self.y=torch.load(f"{diagnosis}/output.pt")
        train_size = int(0.8 * len(self.X))
        self.train_X, self.test_X = self.X[:train_size],self.X[train_size:]
        self.train_y,self.test_y=self.y[:train_size],self.y[train_size:]


    def __getitem__(self, index):
        if self.split_type=="train":
            return self.train_X[index],self.train_y[index]
        return self.test_X[index], self.test_y[index]

    def __len__(self):
        if self.split_type=="train":
            return len(self.train_X)
        return len(self.test_X)


class EvaluationBaseModel(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.linear=nn.Linear(input_size,output_size)

    def forward(self,x):
         out=sigmoid(self.linear(x))
         return out

class EvaluationRNNModel(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,output_size,batch_first=True):
        super().__init__()
        self.rnn=nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                        num_layers=num_layers, batch_first=batch_first)
        self.linear=nn.Linear(hidden_size,output_size)

    def forward(self,X,hidden_state):
        r_out,h_state=self.rnn(X,hidden_state)
        return self.linear(r_out), h_state



if __name__=="__main__":
    with open('sqldata/stats.yaml') as stats:
        stats=yaml.safe_load(stats)

    diagnoses=stats['diagnosis_for_selection']
    torch.seed()
    for diagnosis in diagnoses:
        # combine_timestep_data_rnn_and_dummy_format(diagnosis)

        train_dataset=EvaluationDataset(diagnosis=diagnosis,split_type='train',data_format='dummy')
        test_dataset=EvaluationDataset(diagnosis=diagnosis,split_type='test',data_format='dummy')
        train_loader=DataLoader(train_dataset,shuffle=False,batch_size=1)
        test_loader=DataLoader(test_dataset,shuffle=False,batch_size=1)

        model=EvaluationBaseModel(input_size=902,output_size=1)
        model_rnn=EvaluationRNNModel(input_size=998,hidden_size=1,num_layers=1,output_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion= nn.MSELoss()

        epochs=50
        total_loss=[]
        hidden_state=None
        for epoch in range(epochs):
            for i, (x,y) in enumerate(train_loader):
                y_pred=model(x.float())
                loss=criterion(y_pred.view(1,-1).float(),y.view(1,-1).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print(loss.item())

                # y_pred,hidden_state=model_rnn(x.float(),hidden_state)
                # loss=criterion(y_pred.view(1,-1).float(),y.view(1,-1).float())
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                # hidden_state=list(hidden_state)
                # hidden_state[0]=hidden_state[0].data
                # hidden_state[1]=hidden_state[1].data
                # print(loss.item())

            total_loss.append(loss.item())

        plt.plot(total_loss)
        plt.show()

        #pred
        test_pred=model(test_dataset.X.float())
        test_pred=(test_pred>0.5).float().flatten()
        print("Accuracy",torch.sum(test_pred.detach()==test_dataset.y)/len(test_pred))

