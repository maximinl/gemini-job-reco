SELECT 
  p.subject_id,
  a.hadm_id,
  p.gender AS sex,
  p.anchor_age AS age,
  a.race,
  STRING_AGG(DISTINCT d.icd_code, ', ') AS diagnosis,
  ds.text AS discharge_summary
FROM
  `physionet-data.mimiciv_hosp.patients` p
INNER JOIN
  `physionet-data.mimiciv_hosp.admissions` a ON p.subject_id = a.subject_id
INNER JOIN
  `physionet-data.mimiciv_hosp.diagnoses_icd` d ON a.hadm_id = d.hadm_id
INNER JOIN
  `physionet-data.mimiciv_note.discharge` ds ON a.hadm_id = ds.hadm_id
WHERE
  (d.icd_code LIKE 'F2%' OR d.icd_code LIKE '295%') -- Schizophrenia diagnosis codes
  AND d.seq_num = 1 -- Primary diagnosis only
  AND p.anchor_age <= 40 -- Age filter
GROUP BY
  p.subject_id,
  a.hadm_id,
  p.gender,
  p.anchor_age,
  a.race,
  ds.text
ORDER BY
  p.subject_id, a.hadm_id;
