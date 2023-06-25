import os
import yaml
from model import Graph
import matplotlib.pyplot as plt

dir_=os.path.dirname(__file__)
os.chdir(dir_)

with open('../datasets/sqldata/stats.yaml') as stats:
    stats=yaml.safe_load(stats)

#TODO django management command for running the experiment

stats['diagnosis_for_selection'].remove('ALTERED MENTAL STATUS')
diagnoses=stats['diagnosis_for_selection']
graphs=[]
for diagnosis in diagnoses:
        graph=Graph(diagnosis=diagnosis)()
        graphs.append(graph)


fig,axes=plt.subplots(2,2)
axes_list=[]
axes_list.extend(axes.flatten())
#NOTE we visualize separately so as previous plots do not interfere with existing ax subplot objects
for graph in graphs:
    ax=axes_list.pop(0)
    graph.visualize_cosine_similarities(ax=ax)
fig.supxlabel('Nodes')
fig.supylabel('Similarities')
fig.suptitle('Cosine Similarities for Nodes')
fig.tight_layout()
plt.show()

