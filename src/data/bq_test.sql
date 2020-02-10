
select * from (
select  NUM_ART ,DAT_VTE , QTE_VTE , extract (year from DAT_VTE) as year_of_calendar,
extract (month from DAT_VTE) as month_of_year,
extract (week from DAT_VTE) as week_of_year,

from `dfdp-teradata6y.SalesLmfr.TA001_VTE_REFPETTPJOU`
where  num_art in (

19568094	,
54400801	,
61090575	,
61264070	,
61681963	,
61682530	,
61689075	,
61698560	,
62687464	,
62722520	,
62878284	,
63051016	,
63198485	,
63203875	,
63210196	,
63210280	,
63463540	,
63704060	,
63926856	,
63937433	
	






)
) as A 


