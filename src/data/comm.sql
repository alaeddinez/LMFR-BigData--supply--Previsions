SELECT creation_date as date,
  COUNT(id) as nb_comm
FROM
  `dfdp-teradata6y.CommercialOperationManagementLmfr.T_FAI_ENT_CMDE_GESCO`
WHERE
  creation_date > "2020-01-01"
  AND status <> "CANCELLED"
  
  group by creation_date
  order by creation_date