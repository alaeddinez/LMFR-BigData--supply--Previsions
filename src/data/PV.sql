select dateSession  as date	,sum(nbrFicheProduitVues) as PV from `lmfr-ddp-dwh-prd.traffic_product_card_kpi.traffic_product_cards_kpi`
group by  dateSession
order by dateSession