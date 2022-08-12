import pandas as pd
import numpy as np
import os
from configure import configure
configure()
from mimic3.models import (Admissions, ICUStays,
                           InputEvent_MV, Patients, Prescriptions, Transfers, Services)


def object_field_to_datetime(df,datetime_columns):
    """
    Converts 'object' type .csv fields into 'datetime' fields for
    postgres db population. Mutates original df.

    Args:
        df (pd.DataFrame):
        datetime_columns(List): list of datetime columns names.
    """
    for col in datetime_columns:
            df[col]=pd.to_datetime(df[col], errors='coerce')
    df.replace({np.nan:None},inplace=True)

class PopulateModels:
    """
    Read, process, load models data into postgres.
    """
    def __init__(self) -> None:
        self.main_path = "..\\..\\mimic\\physionet.org\\files\\mimiciii\\1.4"

    def load_patients(self) -> None:
        patients_path_df_path = os.path.join(self.main_path, 'PATIENTS.csv')
        self.patients_df = pd.read_csv(patients_path_df_path)
        self.datetime_columns=['DOB','DOD','DOD_HOSP','DOD_SSN']

        object_field_to_datetime(self.patients_df,self.datetime_columns)
        for _, row in self.patients_df.iterrows():
            try:
                Patients.objects.update_or_create(subject_id=row['SUBJECT_ID'], \
                                                defaults={'row_id': row['ROW_ID'],
                                                            'gender': row['GENDER'],
                                                            'dob': row['DOB'],
                                                            'dod': row['DOD'],
                                                            'dod_hosp':row['DOD_HOSP'],
                                                            'dod_ssn':row['DOD_SSN'],
                                                            'expire_flag':row['EXPIRE_FLAG']})
            except Exception as e:
                print(f"Exception on row_id {row['ROW_ID']}.")
                raise (e)


    def load_admissions(self) -> None:
        admissions_df_path=os.path.join(self.main_path, 'ADMISSIONS.csv')
        self.admissions_df=pd.read_csv(admissions_df_path)
        self.datetime_columns=['ADMITTIME','DISCHTIME','DEATHTIME']
        object_field_to_datetime(self.admissions_df,self.datetime_columns)
        for _, row in self.admissions_df.iterrows():
            try:
                try:
                    patient=Patients.objects.get(subject_id=row['SUBJECT_ID'])
                except Patients.DoesNotExist:
                    print(f"No patient instance with id {row['SUBJECT_ID']}")
                else:
                    Admissions.objects.update_or_create(row_id=row['ROW_ID'],defaults={
                        "subject_id":patient,
                        "hadm_id": row['HADM_ID'],
                        "admittime":row['ADMITTIME'],
                        "dischtime":row['DISCHTIME'],
                        "deathtime":row['DEATHTIME'],
                        "admission_type":row['ADMISSION_TYPE'],
                        "admission_location": row['ADMISSION_LOCATION'],
                        "discharge_location":row['DISCHARGE_LOCATION'],
                        "insurance":row['INSURANCE'],
                        "language":row['LANGUAGE'],
                        "religion":row['RELIGION'],
                        "marital_status":row['MARITAL_STATUS'],
                        "ethnicity":row['ETHNICITY'],
                        "edregtime":row['EDREGTIME'],
                        "edouttime":row['EDOUTTIME'],
                        "diagnosis":row['DIAGNOSIS'],
                        "hospital_expire_flag":row['HOSPITAL_EXPIRE_FLAG'],
                        "has_chartevents_data":row['HAS_CHARTEVENTS_DATA']
                    })

            except Exception as e:
                print(f"Exception on row_id {row['ROW_ID']}.")
                raise (e)


    def load_icustays(self)->None:
        icustays_df_path= os.path.join(self.main_path, 'ICUSTAYS.csv')
        self.icustays_df=pd.read_csv(icustays_df_path)
        self.datetime_columns=['INTIME','OUTTIME']
        object_field_to_datetime(self.icustays_df,self.datetime_columns)
        for _, row in self.icustays_df.iterrows():
            try:
                try:
                    patient=Patients.objects.get(subject_id=row['SUBJECT_ID'])
                except Patients.DoesNotExist:
                    print(f"No patient instance with id {row['SUBJECT_ID']}")
                try:
                    admission=Admissions.objects.get(hadm_id=row['HADM_ID'])
                except Admissions.DoesNotExist:
                    print(f"No admission instance with id {row['HADM_ID']}")
                else:
                    ICUStays.objects.update_or_create(icustay_id=row['ICUSTAY_ID'],defaults=
                                                      {'row_id':row['ROW_ID'],
                                                       'subject_id':patient,
                                                       'hadm_id':admission,
                                                       'dbsource':row['DBSOURCE'],
                                                       'first_careunit':row['FIRST_CAREUNIT'],
                                                       'last_careunit':row['LAST_CAREUNIT'],
                                                       "first_wardid":row['FIRST_WARDID'],
                                                       "last_wardid":row['LAST_WARDID'],
                                                       "intime":row['INTIME'],
                                                       "outtime":row['OUTTIME'],
                                                       "los":row['LOS']
                                                           })
            except Exception as e:
                print(f"Exception on row_id {row['ROW_ID']}.")
                raise (e)

    def load_inputevent_mv(self) -> None:
        inputevent_mv_df_path=os.path.join(self.main_path, 'INPUTEVENTS_MV.csv')
        self.inputevent_mv_df=pd.read_csv(inputevent_mv_df_path)
        for _, row in self.inputevent_mv_df.iterrows():
            try:
                try:
                    patient=Patients.objects.get(subject_id=row['SUBJECT_ID'])
                except Patients.DoesNotExist:
                    print(f"No patient instance with id {row['SUBJECT_ID']}")
                try:
                    admission=Admissions.objects.get(hadm_id=row['HADM_ID'])
                except Admissions.DoesNotExist:
                    print(f"No admission instance with id {row['HADM_ID']}")
                else:
                    InputEvent_MV.objects.update_or_create(row_id=row['ROW_ID'],defaults={
                        "subject_id":patient,
                        "hadm_id": admission,
                        "statusdescription":row['STATUSDESCRIPTION']
                    })

            except Exception as e:
                print(f"Exception on row_id {row['ROW_ID']}.")
                raise (e)


    def load_prescriptions(self) -> None:
        prescriptions_df_path=os.path.join(self.main_path, 'PRESCRIPTIONS.csv')
        self.prescriptions_df=pd.read_csv(prescriptions_df_path)
        self.datetime_columns=['STARTDATE','ENDDATE']
        object_field_to_datetime(self.prescriptions_df,self.datetime_columns)
        for _, row in self.prescriptions_df.iterrows():
            try:
                try:
                    patient=Patients.objects.get(subject_id=row['SUBJECT_ID'])
                except Patients.DoesNotExist:
                    print(f"No patient instance with id {row['SUBJECT_ID']}")
                try:
                    admission=Admissions.objects.get(hadm_id=row['HADM_ID'])
                except Admissions.DoesNotExist:
                    print(f"No admission instance with id {row['HADM_ID']}")
                else:
                    try:
                        icustay=ICUStays.objects.get(icustay_id=row['ICUSTAY_ID'])
                    except ICUStays.DoesNotExist:
                        print(f"No icustay instance with id {row['HADM_ID']}")
                        icustay=None
                      # no icustay is ok, we can create an object with None instance
                    Prescriptions.objects.update_or_create(row_id=row['ROW_ID'],defaults={
                        'subject_id':patient,
                        'hadm_id':admission,
                        'icustay_id':icustay,
                        'startdate':row['STARTDATE'],
                        'enddate':row['ENDDATE'],
                        'drug_type':row['DRUG_TYPE'],
                        'drug':row['DRUG'],
                        'drug_name_poe':row['DRUG_NAME_POE'],
                        'drug_name_generic':row['DRUG_NAME_GENERIC'],
                        'formulary_drug_cd':row['FORMULARY_DRUG_CD'],
                        'gsn':row['GSN'],
                        'ndc':row['NDC'],
                        'prod_strength':row['PROD_STRENGTH'],
                        'dose_val_rx':row['DOSE_VAL_RX'],
                        'dose_unt':row['DOSE_UNIT'],
                        'form_val_disp':row['FORM_VAL_DISP'],
                        'form_unit_disp':row['FORM_UNIT_DISP'],
                        'route':row['ROUTE']
                                            })

            except Exception as e:
                print(f"Exception on row_id {row['ROW_ID']}.")
                raise (e)

    def load_services(self) -> None:
        services_df_path=os.path.join(self.main_path, 'SERVICES.csv')
        self.services_df=pd.read_csv(services_df_path)
        self.datetime_columns=['TRANSFERTIME']
        object_field_to_datetime(self.services_df,self.datetime_columns)
        for _, row in self.services_df.iterrows():
            try:
                try:
                    patient=Patients.objects.get(subject_id=row['SUBJECT_ID'])
                except Patients.DoesNotExist:
                    print(f"No patient instance with id {row['SUBJECT_ID']}")
                try:
                    admission=Admissions.objects.get(hadm_id=row['HADM_ID'])
                except Admissions.DoesNotExist:
                    print(f"No admission instance with id {row['HADM_ID']}")
                else:
                    Services.objects.update_or_create(row_id=row['ROW_ID'],defaults={
                        'subject_id':patient,
                        'hadm_id':admission,
                        'transfertime':row['TRANSFERTIME'],
                        'prev_service':row['PREV_SERVICE'],
                        'curr_service':row['CURR_SERVICE']
                    })

            except Exception as e:
                print(f"Exception on row_id {row['ROW_ID']}.")
                raise (e)


    def load_transfers(self) -> None:
        trainsfers_df_path=os.path.join(self.main_path, 'TRANSFERS.csv')
        self.transfers_df=pd.read_csv(trainsfers_df_path)
        self.datetime_columns=['INTIME','OUTTIME']
        object_field_to_datetime(self.transfers_df,self.datetime_columns)
        for _, row in self.transfers_df.iterrows():
            try:
                try:
                    patient=Patients.objects.get(subject_id=row['SUBJECT_ID'])
                except Patients.DoesNotExist:
                    print(f"No patient instance with id {row['SUBJECT_ID']}")
                try:
                    admission=Admissions.objects.get(hadm_id=row['HADM_ID'])
                except Admissions.DoesNotExist:
                    print(f"No admission instance with id {row['HADM_ID']}")
                else:
                    try:
                        icustay=ICUStays.objects.get(icustay_id=row['ICUSTAY_ID'])
                    except ICUStays.DoesNotExist:
                        print(f"No icustay instance with id {row['ICUSTAY_ID']}")
                        # done because during exception assignment is not reached and
                        # previous valid value is not ovverridden
                        icustay=None
                    # no icustay is ok, we can create an object with None instance
                    Transfers.objects.update_or_create(row_id=row['ROW_ID'],defaults={
                            'subject_id':patient,
                            'hadm_id':admission,
                            'icustay_id':icustay,
                            'dbsource':row['DBSOURCE'],
                            'eventtype':row['EVENTTYPE'],
                            'prev_careunit':row['PREV_CAREUNIT'],
                            'curr_careunit':row['CURR_CAREUNIT'],
                            "prev_wardid":row['PREV_WARDID'],
                            "prev_wardid":row['PREV_WARDID'],
                            "intime":row['INTIME'],
                            "outtime":row['OUTTIME'],
                            "los":row['LOS']
                            })
            except Exception as e:
                    print(f"Exception on row_id {row['ROW_ID']}.")
                    raise (e)

PopulateModels().load_transfers()
