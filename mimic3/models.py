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
	drug_type=models.CharField(max_length=64,null=True,blank=True)
	drug=models.CharField(max_length=64,null=True,blank=True)
	drug_name_poe=models.CharField(max_length=64,null=True,blank=True)
	drug_name_generic=models.CharField(max_length=64,null=True,blank=True)
	formulary_drug_cd=models.CharField(max_length=64,null=True,blank=True)
	gsn=models.DecimalField(max_digits=10,decimal_places=5,null=True,blank=True)
	ndc=models.CharField(max_length=64,null=True,blank=True)
	prod_strength=models.CharField(max_length=64,null=True,blank=True)
	dose_val_rx=models.DecimalField(max_digits=10,decimal_places=5,null=True,blank=True)
	dose_unit=models.CharField(max_length=10,null=True,blank=True)
	form_val_disp=models.DecimalField(max_digits=10,decimal_places=5,null=True,blank=True)
	form_unit_disp=models.CharField(max_length=10,null=True,blank=True)
	route=models.CharField(max_length=10,null=True,blank=True)


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
	statusdescription=models.CharField(max_length=20,null=True,blank=True)

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









