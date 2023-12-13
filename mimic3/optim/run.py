from ucimlrepo import fetch_ucirepo,list_available_datasets
from sklearn.model_selection import train_test_split
from utils import remove_nan,evaluate, evaluate_sklearn,compare
import warnings
warnings.filterwarnings("ignore")
import os

os.chdir("project/mimic3/optim/results/")

def load_evaluate(data,name,remove_categorical=False,**kwargs):
    """
    Load and evaluate custom SCD with and without scaled methods, and sklarn's SCD classifier.
    Compare the methods after saving the results.
    """
    cwd=os.getcwd()
    print("{} shape: {}".format(name,data.data.features.shape))
    X = data.data.features
    y = data.data.targets
    if remove_categorical:
        X=X._get_numeric_data()
    X,y=remove_nan(X,y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    path=os.path.join(cwd,name+'.log')
    no_scale_path=os.path.join(cwd,name+'_no_scale.log')
    sklearn_path=os.path.join(cwd,name+"_sklearn.log")
    evaluate(X_train, X_test, y_train, y_test,save_path=path)
    evaluate(X_train, X_test, y_train, y_test,save_path=no_scale_path,scale=False)
    evaluate_sklearn(X_train, X_test, y_train, y_test,save_path=sklearn_path)
    compare(path,no_scale_path,sklearn_path)

#Loading and Preprocessing
heart_disease = fetch_ucirepo(id=45)
load_evaluate(heart_disease,"heart_disease")

hepatitis = fetch_ucirepo(id=46)
load_evaluate(hepatitis,"hepatitis")

lung_cancer = fetch_ucirepo(id=62)
load_evaluate(lung_cancer,"lung_cancer")

diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
load_evaluate(diabetic_retinopathy_debrecen,"diabetic")

bone_marrow_transplant_children = fetch_ucirepo(id=565)
load_evaluate(bone_marrow_transplant_children,"bone_marrow",remove_categorical=True)

glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
load_evaluate(glioma_grading_clinical_and_mutation_features,"glioma",remove_categorical=True)

heart_failure_clinical_records = fetch_ucirepo(id=519)
load_evaluate(heart_failure_clinical_records,"heart_failure")

ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225)
load_evaluate(ilpd_indian_liver_patient_dataset,"ilpd",remove_categorical=True)
