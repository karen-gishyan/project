from django.db import models


class Admissions(models.Model):
    row_id=models.IntegerField()
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.IntegerField(primary_key=True)
    admittime=models.DateTimeField(null=True,blank=True)
    dischtime=models.DateTimeField(null=True,blank=True)
    deathtime=models.DateTimeField(null=True,blank=True)
    admission_type=models.CharField(max_length=64,null=True,blank=True)
    admission_location=models.CharField(max_length=64,null=True,blank=True)
    discharge_location=models.CharField(max_length=64,null=True,blank=True)
    insurance=models.CharField(max_length=64,null=True,blank=True)
    language=models.CharField(max_length=64,null=True,blank=True)
    religion=models.CharField(max_length=64,null=True,blank=True)
    marital_status=models.CharField(max_length=64,null=True,blank=True)
    ethnicity=models.CharField(max_length=64,null=True,blank=True)
    edregtime=models.CharField(max_length=64,null=True,blank=True)
    edouttime=models.CharField(max_length=64,null=True,blank=True)
    diagnosis=models.CharField(max_length=256,null=True,blank=True)
    hospital_expire_flag=models.BooleanField(null=True,blank=True)
    has_chartevents_data=models.BooleanField(null=True,blank=True)


class Prescriptions(models.Model):
    row_id=models.IntegerField(primary_key=True)
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE)
    icustay_id=models.ForeignKey(to='ICUStays',on_delete=models.CASCADE,null=True,blank=True)
    startdate=models.DateTimeField(null=True,blank=True)
    enddate=models.DateTimeField(null=True,blank=True)
    drug_type=models.CharField(max_length=256,null=True,blank=True)
    drug=models.CharField(max_length=256,null=True,blank=True)
    drug_name_poe=models.CharField(max_length=256,null=True,blank=True)
    drug_name_generic=models.CharField(max_length=256,null=True,blank=True)
    formulary_drug_cd=models.CharField(max_length=256,null=True,blank=True)
    gsn=models.CharField(max_length=128,null=True,blank=True)
    ndc=models.CharField(max_length=128,null=True,blank=True)
    prod_strength=models.CharField(max_length=356,null=True,blank=True)
    dose_val_rx=models.CharField(max_length=256,null=True,blank=True,help_text='Not purely decimal, has different datatypes.')
    dose_unit_rx=models.CharField(max_length=256,null=True,blank=True)
    form_val_disp=models.CharField(max_length=256,null=True,blank=True,help_text='Not purely decimal, has different datatypes.')
    form_unit_disp=models.CharField(max_length=256,null=True,blank=True)
    route=models.CharField(max_length=256,null=True,blank=True)


class ICUStays(models.Model):
    row_id=models.IntegerField()
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE)
    icustay_id=models.PositiveIntegerField(primary_key=True)
    dbsource=models.CharField(max_length=10,null=True,blank=True)
    first_careunit=models.CharField(max_length=10,null=True,blank=True)
    last_careunit=models.CharField(max_length=10,null=True,blank=True)
    first_wardid=models.IntegerField(null=True,blank=True)
    last_wardid=models.IntegerField(null=True,blank=True)
    intime=models.DateTimeField(null=True,blank=True)
    outtime=models.DateTimeField(null=True,blank=True)
    los=models.DecimalField(max_digits=10,decimal_places=5,null=True,blank=True)


class InputEvent_MV(models.Model):
    row_id=models.IntegerField(primary_key=True)
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE)
    patient_weight=models.DecimalField(max_digits=5,decimal_places=2,null=True,blank=True)


class Patients(models.Model):
    row_id=models.IntegerField()
    subject_id=models.PositiveIntegerField(primary_key=True)
    gender=models.CharField(max_length=2,null=True,blank=True)
    dob=models.DateTimeField(null=True,blank=True)
    dod=models.DateTimeField(null=True,blank=True)
    dod_hosp=models.DateTimeField(null=True,blank=True)
    dod_ssn=models.DateTimeField(null=True,blank=True)
    expire_flag=models.SmallIntegerField(null=True,blank=True)


class Services(models.Model):
    row_id=models.IntegerField(primary_key=True)
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE)
    transfertime=models.DateTimeField(null=True,blank=True)
    prev_service=models.CharField(max_length=20,null=True,blank=True)
    curr_service=models.CharField(max_length=20,null=True,blank=True)


class Transfers(models.Model):
    row_id=models.IntegerField(primary_key=True)
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE)
    icustay_id=models.ForeignKey(to='ICUStays',on_delete=models.CASCADE,null=True,blank=True)
    dbsource=models.CharField(max_length=10,null=True,blank=True)
    eventtype=models.CharField(max_length=10,null=True,blank=True)
    prev_careunit=models.CharField(max_length=10,null=True,blank=True)
    curr_careunit=models.CharField(max_length=10,null=True,blank=True)
    prev_wardid=models.SmallIntegerField(null=True,blank=True)
    curr_wardid=models.SmallIntegerField(null=True,blank=True)
    intime=models.DateTimeField(null=True,blank=True)
    outtime=models.DateTimeField(null=True,blank=True)
    los=models.DecimalField(max_digits=10,decimal_places=5,null=True,blank=True)


class D_Items(models.Model):
    row_id=models.IntegerField()
    itemid=models.IntegerField(primary_key=True)
    label=models.CharField(max_length=100,null=True, blank=True)
    abbreviation=models.CharField(max_length=100,null=True, blank=True)
    dbsource=models.CharField(max_length=100,null=True, blank=True)
    linksto=models.CharField(max_length=100,null=True, blank=True)
    category=models.CharField(max_length=100,null=True, blank=True)
    unitname=models.CharField(max_length=100,null=True, blank=True)
    param_type=models.CharField(max_length=100,null=True, blank=True)
    conceptid=models.CharField(max_length=100,null=True, blank=True)


class D_LabItems(models.Model):
    row_id=models.IntegerField()
    itemid=models.IntegerField(primary_key=True)
    label=models.CharField(max_length=100,null=True, blank=True)
    fluid=models.CharField(max_length=100,null=True, blank=True)
    category=models.CharField(max_length=100,null=True, blank=True)
    loinc_code=models.CharField(max_length=100,null=True, blank=True)


class ChartEvents(models.Model):
    row_id=models.IntegerField(primary_key=True)
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE)
    icustay_id=models.ForeignKey(to='ICUStays',on_delete=models.CASCADE,null=True,blank=True)
    itemid=models.ForeignKey(to='D_Items',on_delete=models.CASCADE,to_field='itemid')
    charttime=models.DateTimeField(null=True,blank=True)
    storetime=models.DateTimeField(null=True,blank=True)
    cgid=models.PositiveIntegerField(null=True,blank=True)
    value=models.CharField(max_length=100,null=True,blank=True)
    valuenum=models.DecimalField(max_digits=100,decimal_places=3,null=True,blank=True)
    valueuom=models.CharField(max_length=100,null=True,blank=True)
    warning=models.PositiveIntegerField(null=True,blank=True)
    error=models.PositiveIntegerField(null=True,blank=True)
    resultstatus=models.CharField(max_length=100,null=True,blank=True)
    stopped=models.CharField(max_length=100,null=True, blank=True)


class LabEvents(models.Model):
    row_id=models.IntegerField(primary_key=True)
    subject_id=models.ForeignKey(to='Patients', on_delete=models.CASCADE, to_field='subject_id')
    hadm_id=models.ForeignKey(to='Admissions',on_delete=models.CASCADE,null=True)
    itemid=models.ForeignKey(to='D_LabItems',on_delete=models.CASCADE,to_field='itemid')
    charttime=models.DateTimeField(null=True,blank=True)
    value=models.CharField(max_length=100,null=True,blank=True)
    valuenum=models.DecimalField(max_digits=100,decimal_places=3,null=True,blank=True)
    valueuom=models.CharField(max_length=100,null=True,blank=True)
    flag=models.CharField(max_length=100,null=True,blank=True)






