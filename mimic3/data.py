from configure import configure
configure()
import os
import pandas as pd


def process_and_combine_datasets():
    """
    We get each unique admissions last 10 'Drug and 'Heart Rate' feature values ordered by
    the care unit. In other words, what are the patient features based on the fact that they
    have been admitted to a given care unit.
    Returns:
        pf.DataFrame: Combined/concatinated dataframe.
    """
    dir_=os.path.dirname(__file__)
    dir_=os.path.join(dir_,'datasets')
    os.chdir(dir_)

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
        # if concat did not append the same columns for each concact operations
        # would have been a better solution.
        care_units_with_admissions_copy.loc[lower:upper,list(df_admissions.columns)]=df_admissions.values

    care_units_with_admissions_copy.dropna(inplace=True)
    care_units_with_admissions_copy.to_csv('combined.csv')

    return care_units_with_admissions_copy.dropna(inplace=True)

