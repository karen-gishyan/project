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


zSELECT mimic3_prescriptions.hadm_id_id, mimic3_prescriptions.drug,mimic3_prescriptions.dose_val_rx,mimic3_prescriptions.startdate as prescription_start,mimic3_prescriptions.enddate as prescription_end
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
