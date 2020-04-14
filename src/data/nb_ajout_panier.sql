select sessionDate as date , sum(AddToCart) nb_ajout_panier from `big-data-dev-lmfr.COVID19_Tunnel_Commande_DEV.command_tunnel_PDP_Vues_AddCart`
group by sessionDate
order by sessionDate