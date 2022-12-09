import os
import torch
import numpy as np
from itertools import product
from torch.nn.functional import pad

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
