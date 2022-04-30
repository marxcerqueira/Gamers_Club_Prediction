select
    *
from tb_players_medalha as t1

left join tb_medalha as t2
on t1.idMedal = t2.idMedal

where dtCreatedAt < dtExpiration
and dtCreatedAt < '2022-02-01'
and coalesce(dtRemove, dtExpiration) > date('2022-02-01', '-30 day' )
