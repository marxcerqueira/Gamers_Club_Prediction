-- left join

/*

select
    t1.idPlayer,
    t1.descCountry,
    count(distinct t2.idLobbyGame) as qtyLobby,
    sum(t2.qtKill) as totalQtyKills,
    sum(t2.qtHs) as totalQtyHS,
    count(distinct t2.descMapName)

from tb_players as t1

left join tb_lobby_stats_player as t2
on t1.idPlayer = t2.idPlayer

group by t1.idPlayer

*/

-- left where
-- mapa mais jogado no brasil

/*
select
    t2.descMapName,
    count(DISTINCT t2.idLobbyGame) as qtyGames

from tb_players as t1

left join tb_lobby_stats_player as t2
on t1.idPlayer = t2.idPlayer

where t1.descCountry = 'br'
or t1.descCountry = 'ar'

group by t2.descMapName
order by count(DISTINCT t2.idLobbyGame) desc
*/

-- qty de partidas por mapa por país

select
    t1.descCountry,
    t2.descMapName,
    count(distinct t2.idLobbyGame) as qtyLobby

from tb_players as t1

left join tb_lobby_stats_player as t2
on t1.idPlayer = t2.idPlayer

group by t1.descCountry, t2.descMapName
order by t1.descCountry ,count(distinct t2.idLobbyGame) desc