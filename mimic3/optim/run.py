from ucimlrepo import fetch_ucirepo,list_available_datasets
from sklearn.model_selection import train_test_split
from utils import remove_nan,evaluate, compare
import warnings
warnings.filterwarnings("ignore")


#Loading and Preprocessing
heart_disease = fetch_ucirepo(id=45)
print("heart_disease dataset shape {}".format(heart_disease.data.features.shape))
X = heart_disease.data.features
y = heart_disease.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/heart_disease.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/heart_disease_no_scale.log',scale=False)
compare('project/mimic3/optim/results/heart_disease.log','project/mimic3/optim/results/heart_disease_no_scale.log')

hepatitis = fetch_ucirepo(id=46)
print("hepatitis dataset shape {}".format(hepatitis.data.features.shape))
X = hepatitis.data.features
y = hepatitis.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/hepatitis.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/hepatitis_no_scale.log',scale=False)
compare('project/mimic3/optim/results/hepatitis.log','project/mimic3/optim/results/hepatitis_no_scale.log')


lung_cancer = fetch_ucirepo(id=62)
print("lung_cancer dataset shape {}".format(lung_cancer.data.features.shape))
X = lung_cancer.data.features
y = lung_cancer.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/lung_cancer.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/lung_cancer_no_scale.log',scale=False)
compare('project/mimic3/optim/results/lung_cancer.log','project/mimic3/optim/results/lung_cancer_no_scale.log')


diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
print("diabetic_retinopathy_debrecen dataset shape {}".format(diabetic_retinopathy_debrecen.data.features.shape))
X = diabetic_retinopathy_debrecen.data.features
y = diabetic_retinopathy_debrecen.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/diabetic.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/diabetic_no_scale.log',scale=False)
compare('project/mimic3/optim/results/diabetic.log','project/mimic3/optim/results/diabetic_no_scale.log')


bone_marrow_transplant_children = fetch_ucirepo(id=565)
print("bone_marrow_transplant_children dataset shape {}".format(bone_marrow_transplant_children.data.features.shape))
X = bone_marrow_transplant_children.data.features
y = bone_marrow_transplant_children.data.targets
X=X._get_numeric_data()
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/bone_marrow.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/bone_marrow_no_scale.log',scale=False)
compare('project/mimic3/optim/results/bone_marrow.log','project/mimic3/optim/results/bone_marrow_no_scale.log')


glioma_grading_clinical_and_mutation_features = fetch_ucirepo(id=759)
print("glioma_grading_clinical_and_mutation_features dataset shape {}".format(glioma_grading_clinical_and_mutation_features.data.features.shape))
X = glioma_grading_clinical_and_mutation_features.data.features
y = glioma_grading_clinical_and_mutation_features.data.targets
X=X._get_numeric_data()
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/glioma.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/glioma_no_scale.log',scale=False)
compare('project/mimic3/optim/results/glioma.log','project/mimic3/optim/results/glioma_no_scale.log')

heart_failure_clinical_records = fetch_ucirepo(id=519)
print("heart_failure_clinical_records dataset shape {}".format(heart_failure_clinical_records.data.features.shape))
X = heart_failure_clinical_records.data.features
y = heart_failure_clinical_records.data.targets
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/heart_failure.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/heart_failure_no_scale.log',scale=False)
compare('project/mimic3/optim/results/heart_failure.log','project/mimic3/optim/results/heart_failure_no_scale.log')


ilpd_indian_liver_patient_dataset = fetch_ucirepo(id=225)
print("ilpd_indian_liver_patient_dataset dataset shape {}".format(ilpd_indian_liver_patient_dataset.data.features.shape))
X = ilpd_indian_liver_patient_dataset.data.features
y = ilpd_indian_liver_patient_dataset.data.targets
X=X._get_numeric_data()
X,y=remove_nan(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/ilpd.log')
evaluate(X_train, X_test, y_train, y_test,save_path='project/mimic3/optim/results/ilpd_no_scale.log',scale=False)
compare('project/mimic3/optim/results/ilpd.log','project/mimic3/optim/results/ilpd_no_scale.log')
