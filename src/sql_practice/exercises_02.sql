
/* 
1) QUAL O PLAYER COM MAIOR TAXA MÉDIA DE HS? E O MENOR?

select
    idPlayer,
    count(idLobbyGame) as qtyLobbyGame,
    sum(qtKill) as totalKills,
    sum(qtHs) as totalHs,
    round(100.0*sum(qtHs)/sum(qtKill),2) as txHS

from tb_lobby_stats_player
group by idPlayer
order by txHS desc;

*/

/* 
2) QUAL PAÍS QUE POSSUI MAIS JOGADORES?

select
    count(DISTINCT idPlayer) as qtyPlayers,
    descCountry

from tb_players
group by descCountry
order by qtyPlayers desc;

*/

/* 
3) QUAL MAPA MAIS JOGADO?

select
    count(distinct idLobbyGame) as qtyGames,
    count(*) as qtyGamesPlayers,
    descMapName

from tb_lobby_stats_player 
group by descMapName
order by qtyGames desc
;

*/

/* DAP: Daily Active Player
4) QUAL HISTÓRICO DE DAP?

select 
    date(dtCreatedAt) as dtDAP,
    count(distinct idPlayer) AS DAP    

from tb_lobby_stats_player 
group by date(dtCreatedAt)
order by date(dtCreatedAt)

limit 100;

*/


/* 
5) QUAL DIA DE MAIOR DAP?
*/

select 
    date(dtCreatedAt) as dtDAP,
    count(distinct idPlayer) AS DAP    

from tb_lobby_stats_player 
group by date(dtCreatedAt)
order by count(distinct idPlayer) desc

limit 100;