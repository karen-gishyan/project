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