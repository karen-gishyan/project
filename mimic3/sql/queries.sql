SELECT subject_id_id,count(subject_id_id) AS patient_count
FROM mimic3_icustays
GROUP BY subject_id_id
HAVING count(subject_id_id)>1
ORDER BY count(subject_id_id) DESC;

/* One admission may be connected to more than 1  ICU_stay instances.*/
SELECT hadm_id_id,count(hadm_id_id) AS admission_count
FROM mimic3_icustays
GROUP BY hadm_id_id
HAVING count(hadm_id_id)>1
ORDER BY count(subject_id_id) DESC;


SELECT mimic3_prescriptions.hadm_id_id, mimic3_prescriptions.drug,mimic3_prescriptions.dose_val_rx,mimic3_prescriptions.startdate as prescription_start,mimic3_prescriptions.enddate as prescription_end
into prescriptions_subset
from mimic3_prescriptions limit 10000;

select prescriptions_subset.hadm_id_id,
prescriptions_subset.drug,
prescriptions_subset.dose_val_rx,
prescriptions_subset.prescription_start,
prescriptions_subset.prescription_end,
mimic3_transfers.eventtype,
mimic3_transfers.prev_careunit,
mimic3_transfers.curr_careunit,
mimic3_transfers.intime,
mimic3_transfers.outtime,
mimic3_chartevents.value,
mimic3_chartevents.charttime,
mimic3_chartevents.storetime,
mimic3_d_items.label
from prescriptions_subset
Inner JOIN mimic3_transfers
on mimic3_transfers.hadm_id_id=prescriptions_subset.hadm_id_id
INNER Join mimic3_chartevents
on prescriptions_subset.hadm_id_id=mimic3_chartevents.hadm_id_id
INNER Join mimic3_d_items
on mimic3_chartevents.itemid_id=mimic3_d_items.itemid;


select mimic3_icustays.first_careunit,
prescriptions_subset.hadm_id_id
into temp_db
from prescriptions_subset
Inner JOIN mimic3_icustays
on mimic3_icustays.hadm_id_id=prescriptions_subset.hadm_id_id
GROUP by mimic3_icustays.first_careunit, prescriptions_subset.hadm_id_id;

select a.first_careunit,a.hadm_id_id,b.drug
from temp_db a
Inner join mimic3_prescriptions b
on a.hadm_id_id=b.hadm_id_id;

--Get the list of admissions partiotioned by ther first care unit, the last 10 medications and other health related features.

-- First partition by admission that have at least 10 drugs(10 and more).
-- Then filter by row_id_count with <=10. The first condition guarantees that each admisison has exactly 10 drugs.
select * from
(select * from
(select mimic3_icustays.first_careunit,
mimic3_icustays.last_careunit,
mimic3_prescriptions.hadm_id_id,
mimic3_prescriptions.drug,
mimic3_prescriptions.startdate,mimic3_prescriptions.enddate,
mimic3_admissions.discharge_location,
count(*) over (PARTITION by  mimic3_prescriptions.hadm_id_id order by  mimic3_prescriptions.hadm_id_id) as id_count,
row_number() over (PARTITION by  mimic3_prescriptions.hadm_id_id order by  mimic3_prescriptions.hadm_id_id) as row_id_count
from mimic3_prescriptions
INNER join
mimic3_icustays on mimic3_icustays.hadm_id_id=mimic3_prescriptions.hadm_id_id
Inner Join
mimic3_admissions on mimic3_admissions.hadm_id=mimic3_prescriptions.hadm_id_id) AS inner_table
WHERE id_count>=10
Order by first_careunit,hadm_id_id,startdate) as inner_tabel_2
where row_id_count<=10;

-- Get all drugs per admission that appear ordered by admission, drug and date.
-- First make sure that more than 10 obs exist per admission, so as exactly 10 are selected for each.
select * from
(select * from
(select mimic3_chartevents.hadm_id_id,mimic3_d_items.itemid,mimic3_d_items.label, mimic3_chartevents.value,
mimic3_chartevents.charttime,
count(*) over (PARTITION by  mimic3_chartevents.hadm_id_id,mimic3_d_items.itemid) as partition_count,
row_number() over (PARTITION by  mimic3_chartevents.hadm_id_id,mimic3_d_items.itemid) as row_partition_count
from mimic3_chartevents
inner join mimic3_d_items
on mimic3_chartevents.itemid_id=mimic3_d_items.itemid) as inner_table
where partition_count>=10
order by hadm_id_id,itemid,charttime) as innet_table_2
where row_partition_count<=10;

-- check the overlap between two tables.
select Distinct df1.hadm_id_id from df1
where df1.hadm_id_id in
(select Distinct df2.hadm_id_id from df2);
