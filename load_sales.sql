select * from (
select cast(NUM_ART as int) as NUM_ART ,DAT_VTE , QTE_VTE  from BV_PROD_001_DWHVIEW.T_AGG_AGGVTE_REFPETTPJOU 
where  num_art in (select  distinct NUM_ART  from BV_PROD_001_PBSDBS.TA001_RAR_ARTMAG 
where NUM_FOUCOM  in (6276,204015))
) as A 
left join 
(
 select calendar_date as DAT_VTE ,  week_of_year , month_of_quarter, month_of_year,quarter_of_year,year_of_calendar from SYS_CALENDAR.CALENDAR
) as B 
ON  A.DAT_VTE = B.DAT_VTE