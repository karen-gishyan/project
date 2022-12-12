import os
import torch
import numpy as np
from itertools import product
from torch.nn.functional import pad
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from evaluate import DataSet, EvaluationModel

def remove_file_from_subdirectories(file_name):
    """
    Find the file in the subdirectories of the root folder.
    Takes user input before deleting the file.
    """
    dir_=os.path.dirname(__file__)
    os.chdir(dir_)
    for root,dirs, files in os.walk("."):
        for file in files:
            if file==file_name:
                path=os.path.join(root,file)
                print(path)
                user_input=input("should the file be removed?:")
                if str(user_input)=='yes':
                    os.remove(path)

def combine_drug_sequences(diagnosis,dir_name,method=None):
    """
     For each testing instance we generate 125 drug sequences.
    """
    cd=os.getcwd()
    sequences=[]
    for t in [1,2,3]:
        if method:
            path=os.path.join(cd,diagnosis,dir_name,method,f"t{t}","drug_sequences.pt")
        else:
            path=os.path.join(cd,diagnosis,dir_name,f"t{t}","drug_sequences.pt")
        tensor=torch.load(path)
        sequences.append(tensor)

    row_shapes=[i.shape[1] for i in sequences]
    col_shapes=[i.shape[2] for i in sequences]
    max_rows,max_cols=max(row_shapes),max(col_shapes)
    # drugs in the same t are padded to have the same number of rows and columns,
    # however both rows and columns can differ across ts depending on the method.
    # we thus pad with -1 the difference between row and maximum row, col and max col.
    for i in range(len(sequences)):
        n_rows=sequences[i].shape[1]
        n_cols=sequences[i].shape[2]
        diff_rows=max_rows-n_rows
        diff_cols=max_cols-n_cols
        # padding tuple logic is left,right, up, down, thus we pad right and down
        sequences[i]=pad(sequences[i],(0,diff_cols,0,diff_rows),value=-1)

    t1_sequence,t2_sequence,t3_sequence=sequences
    batch_size=t1_sequence.shape[0]
    # should be the same for all columns
    n_cols=t1_sequence.shape[2]
    combinations=[]
    for batch in range(batch_size):
        concat=list(product(t1_sequence[batch].tolist(), t2_sequence[batch].tolist(),t3_sequence[batch].tolist()))
        concat=torch.Tensor(np.array(concat).reshape((-1,n_cols*3)))
        combinations.append(concat)
    final_tensor=torch.cat((*combinations,),dim=0)
    if method:
        torch.save(final_tensor,os.path.join(cd,diagnosis,dir_name,method,"combined_drugs.pt"))
    else:
        torch.save(final_tensor,os.path.join(cd,diagnosis,dir_name,"combined_drugs.pt"))


    # increase the amount of test size based on the expansion of product
    test_output=torch.load(f'{diagnosis}/test_output.pt')
    expand_count=final_tensor.shape[0]/test_output.shape[0]
    test_output_expanded=torch.Tensor(np.repeat(test_output.numpy(),expand_count))
    #test output is the same for all methods, but test_output_expanded can differ
    # across methods
    if method:
         path=f'{diagnosis}/{dir_name}/{method}/test_output_expanded.pt'
    else:
        path=f'{diagnosis}/{dir_name}/test_output_expanded.pt'
    assert final_tensor.shape[0]==test_output_expanded.shape[0], \
     f"Number of rows do not match {diagnosis},{dir_name},{method}"
    torch.save(test_output_expanded,path)

def train_individual(diagnosis, dirname,method=None):
    dataset=DataSet(diagnosis,dirname,method)
    input_size=dataset.train_X.shape[1]
    output_size=dataset.train_y.shape[1]

    loader=DataLoader(dataset,shuffle=True,batch_size=50)
    model=EvaluationModel(input_size=input_size,output_size=output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    criterion= nn.MSELoss()

    epochs=150
    total_loss=[]
    for _ in range(epochs):
        for _, (x,y) in enumerate(loader):
            y_pred=model(x.float())
            loss=criterion(y_pred.float(),y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss.append(loss.item())

    plt.plot(total_loss)
    title=f"Diagnosis:{diagnosis}, Type:{dirname}, Method:{method}" if method else \
        f"Diagnosis:{diagnosis}, Type:{dirname}"
    plt.title(title)
    # plt.show()

    #pred
    test_X=dataset.test_X
    test_y=dataset.test_y
    print(f"{diagnosis},{dirname},{method}")
    if torch.all(test_X==0):
        print("All test inputs are zero.")
    if torch.all(test_y==0):
        print("All test outputs are zero.")
    test_pred=model(test_X.float())
    test_pred=(test_pred>0.5).float().flatten()
    print("Accuracy",torch.sum(test_pred.detach()==test_y.flatten())/len(test_pred))
