with
SoldOnline as (
select distinct cast (sku as string ) productId,dateSession as dat_jou, boutonPanier as FLAG_Online from 
`big-data-dev-lmfr.supply_test.bouton_ajout_panier_livraison_tout`)
,
pageviews as (
select productSKU,dateSession as date	,sum(nbrFicheProduitVues) as PV from `lmfr-ddp-dwh-prd.traffic_product_card_kpi.traffic_product_cards_kpi`
where dateSession	 > "2020-03-14" and nbrFicheProduitVues is not null
group by productSKU,dateSession
order by dateSession
),
final_table_1803 as (
select productsku,	date	,PV	,	FLAG_Online ,NUM_RAY, NUM_SRAY, NUM_TYP,NUM_STYP,
TOP_AVSART	, DAT_AVSART,NUM_TYPREAPPROPRECO	
from (
select  productsku,	date	,PV	,	FLAG_Online	
from (select productsku, date, PV from pageviews ) as A
left join 
(select * from SoldOnline) as B
ON A.productsku = B.productId  
and A.date = B.dat_jou)
as  A1
left join 
(SELECT CAST (NUM_ART as string) NUM_ART, NUM_RAY, NUM_SRAY, NUM_TYP,NUM_STYP , 
TOP_AVSART	, DAT_AVSART,NUM_TYPREAPPROPRECO	

FROM `dfdp-teradata6y.ProductCatalogLmfr.TA001_RAR_ART` ) as nomen
ON productsku = NUM_ART)
#where FLAG_Online is null)


####
select  productsku,	date	,PV	,	
case when FLAG_Online = "oui" then "oui"
else "non"

end as FLAG_Online,NUM_RAY, NUM_SRAY, NUM_TYP,NUM_STYP,
TOP_AVSART	, DAT_AVSART,NUM_TYPREAPPROPRECO , flag_380_TYP2, ref_prio from (
select productsku,	date	,PV	,	FLAG_Online ,NUM_RAY, NUM_SRAY, NUM_TYP,NUM_STYP,
TOP_AVSART	, DAT_AVSART,NUM_TYPREAPPROPRECO , flag_380_TYP2	 from (
select * from final_table_1803 ) as A
left join
( select distinct CAST( NUM_ART as string) NUM_ART , 1 as flag_380_TYP2 from `dfdp-teradata6y.ProductCatalogLmfr.TA001_RAR_ARTMAG`
where NUM_ETT = 380 and NUM_TYPREAPPRO	 = 2) as B
on A.productsku = B.NUM_ART
) as A
left join 
(select  cast(int64_field_0 as string) int64_field_0, "1" as ref_prio 	 from `big-data-dev-lmfr.supply_test.ref_prio`) as B
ON A.productsku = B.int64_field_0
order by PV desc