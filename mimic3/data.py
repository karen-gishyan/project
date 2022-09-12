from configure import configure
configure()
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


dir_=os.path.dirname(__file__)
dir_=os.path.join(dir_,'datasets')
os.chdir(dir_)

def process_and_combine_datasets():
    """
    We get each unique admissions last 10 'Drug and 'Heart Rate' feature values ordered by
    the care unit. In other words, what are the patient features based on the fact that they
    have been admitted to a given care unit.
    Returns:
        pf.DataFrame: Combined/concatinated dataframe.
    """

    care_units_with_admissions=pd.read_csv('care_units_with_admisisons_and_drugs.csv',usecols=[i for i in range(7)])
    care_units_with_admissions.rename(columns={'hadm_id_id':'hadm_id'},inplace=True)
    # copied for joining purposes
    care_units_with_admissions_copy=care_units_with_admissions.copy().reset_index()
    admissions_and_patient_features=pd.read_csv('admissions_and_patient_features.csv',usecols=[i for i in range(5)])
    admissions_and_patient_features=admissions_and_patient_features[admissions_and_patient_features.itemid==220045]

    care_units_admissions_col=list(care_units_with_admissions.hadm_id.unique())
    admissions_admissions_col=list(admissions_and_patient_features.hadm_id_id.unique())
    admisisons_intersection=sorted(list(set(care_units_admissions_col) & set(admissions_admissions_col)))

    care_units_with_admissions_copy=care_units_with_admissions_copy.reindex(columns=list(care_units_with_admissions.columns)
                                            +list(admissions_and_patient_features.columns))
    for admission_id in care_units_admissions_col:
        if not admission_id in admisisons_intersection:
            continue
        subset_index=care_units_with_admissions[care_units_with_admissions.hadm_id==admission_id].index
        df_admissions=admissions_and_patient_features[(admissions_and_patient_features.hadm_id_id==admission_id)
                                                      &(admissions_and_patient_features.itemid==220045)]

        df_admissions.index=subset_index
        lower,upper=list(subset_index)[0],list(subset_index)[-1]
        # copy values from the second df to the first df
        # if concat did not append the same columns for each concact operations it
        # would have been a better solution.
        care_units_with_admissions_copy.loc[lower:upper,list(df_admissions.columns)]=df_admissions.values

    care_units_with_admissions_copy.dropna(inplace=True)
    care_units_with_admissions_copy.to_csv('combined.csv')

    return care_units_with_admissions_copy.dropna(inplace=True)


def construct_pytorch_dataset():
    df=pd.read_csv('combined.csv')
    # remove some columns then select only categorical columns
    df_categorical=df.loc[:,~df.columns.isin(['startdate','enddate','charttime'])].\
        select_dtypes(include=['object'])

    # encode labels into a single column
    for col in df_categorical.columns:
        df_categorical[col]=LabelEncoder().fit_transform(df_categorical[col])
    df_categorical['value']=df['value']
    df_categorical['hadm_id']=df['hadm_id']

    unique_admissions=df_categorical.hadm_id.unique()
    timestep1_df,timestep2_df, timestep3_df=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for adm in unique_admissions:
        iter_df=df_categorical[df_categorical.hadm_id==adm]
        timestep1_df=pd.concat([timestep1_df,iter_df[0:3]])
        timestep2_df=pd.concat([timestep2_df,iter_df[3:6]])
        timestep3_df=pd.concat([timestep3_df,iter_df[6:]])


    timestep1_df.to_csv("pytorch_t1_df.csv",index=False)
    timestep2_df.to_csv("pytorch_t2_df.csv",index=False)
    timestep3_df.to_csv("pytorch_t3_df.csv",index=False)
    print('Done')

