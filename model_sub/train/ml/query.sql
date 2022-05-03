drop table if exists tb_abt_sub;
create table tb_abt_sub as 

-- business problem
-- Qual a probabilidade de uma pessoa assinar a GC nos próximos 15 dias?

with

tb_subs as (

    select
        t1.idPlayer,
        t1.idMedal,
        t1.dtCreatedAt,
        t1.dtExpiration,
        t1.dtRemove

    from tb_players_medalha as t1

    left join tb_medalha as t2
    on t1.idMedal = t2.idMedal

    where t2.descMedal in ('Membro Premium', 'Membro Plus')
    and coalesce(t1.dtExpiration, date('now')) > t1.dtCreatedAt

)

select

    t1.*,
    case when t2.idMedal is null then 0 else 1 end as flagSub

from tb_book_players as t1

left join tb_subs as t2
on t1.idPlayer = t2.idPlayer

-- pegando em 15 dias quem assinou e quem não assinou
and t1.dtReff < t2.dtCreatedAt
and t2.dtCreatedAt < date(t1.dtReff, '+15 day')
and t1.dtReff < date('2022-02-01', '-15 day')

where AssinaturaAtiva = 0
;