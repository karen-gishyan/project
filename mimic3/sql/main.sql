SELECT * FROM
(SELECT * FROM
(SELECT mimic3_prescriptions.hadm_id_id,
mimic3_admissions.admittime,
mimic3_admissions.dischtime,
mimic3_admissions.diagnosis,
mimic3_icustays.first_careunit,
mimic3_icustays.last_careunit,
mimic3_prescriptions.drug,
mimic3_prescriptions.row_id,
mimic3_prescriptions.startdate,mimic3_prescriptions.enddate,

mimic3_admissions.discharge_location,
count(*) OVER (PARTITION BY  mimic3_prescriptions.hadm_id_id ORDER BY  mimic3_prescriptions.hadm_id_id) AS id_count,
row_number() OVER (PARTITION BY  mimic3_prescriptions.hadm_id_id ORDER BY  mimic3_prescriptions.hadm_id_id) AS row_id_count
FROM mimic3_prescriptions
INNER JOIN
mimic3_icustays ON mimic3_icustays.hadm_id_id=mimic3_prescriptions.hadm_id_id
INNER JOIN
mimic3_admissions ON mimic3_admissions.hadm_id=mimic3_prescriptions.hadm_id_id) AS inner_table
WHERE id_count>=10
ORDER BY first_careunit,hadm_id_id,row_id) AS inner_tabel_2;

--
--
SELECT * FROM
(SELECT * FROM
(SELECT mimic3_chartevents.hadm_id_id,mimic3_d_items.itemid,mimic3_d_items.label, mimic3_chartevents.value,
mimic3_admissions.diagnosis,
mimic3_chartevents.charttime, mimic3_admissions.admittime,
mimic3_admissions.dischtime,
count(*) OVER (PARTITION BY  mimic3_chartevents.hadm_id_id,mimic3_d_items.itemid) AS partition_count,
row_number() OVER (PARTITION BY  mimic3_chartevents.hadm_id_id,mimic3_d_items.itemid) AS row_partition_count
FROM mimic3_chartevents
INNER JOIN mimic3_d_items
ON mimic3_chartevents.itemid_id=mimic3_d_items.itemid
INNER JOIN mimic3_admissions
ON mimic3_chartevents.hadm_id_id=mimic3_admissions.hadm_id
) AS inner_table
WHERE partition_count>=10
ORDER BY hadm_id_id,itemid,charttime) AS innet_table_2
WHERE partition_count<=300;