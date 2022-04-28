/* 
1) QUAL O WINRATE DOS ASSINANTES PLUS VS PREMIUM

select 
    * 

from tb_players_medalha as t1

left join tb_lobby_stats_player as t2
on t1.idPlayer = t2.idPlayer

where descMedal IN ('Membro Plus', 'Membro Premium')

-- precisamos de subquery para resolver

select
    t1.idPlayer,
    t1.idLobbyGame,
    t1.flWinner,
    t2.idMedal,
    t2.flActive,
    t3.idMedal,
    t3.descMedal

from tb_lobby_stats_player as t1

left join tb_players_medalha as t2
on t1.idPlayer = t2.idPlayer

left join tb_medalha as t3
on t2.idMedal = t3.idMedal

where t3.descMedal in ('Membro Plus', 'Membro Premium')

limit 50
*/

/* 
2) DE QUAL PAÍS É O JOGADOR QUE TEM MAIOR TX DE HS
select
    t1.idPlayer,
    t2.descCountry,
    avg(100.0*t1.qtHs/t1.qtKill) as txHs,
    count(DISTINCT idLobbyGame),
    sum(qtKill)

from tb_lobby_stats_player as t1

left join tb_players as t2
on t1.idPlayer = t2.idPlayer

group by t1.idPlayer,t2.descCountry

having count(DISTINCT idLobbyGame) >= 5

order by avg(100.0*t1.qtHs/t1.qtKill) desc
*/

/* 
3) QUAL MAPA MAIS JOGADO PELOS ARGENTINOS

SELECT
    t1.descMapName,
    t2.descCountry,
    count(distinct t1.idPlayer),
    count(distinct t1.idLobbyGame)
    

from tb_lobby_stats_player as t1

left join tb_players as t2
on t1.idPlayer = t2.idPlayer

group by t2.descCountry, t1.descMapName

having t2.descCountry = 'ar'

order by count(distinct t1.idLobbyGame) desc

*/

/* 
4) QUAL A TAXA DE HS DA TRIBO
-- tbm precisa de subquery

select
    t1.idPlayer,
    t1.idLobbyGame,
    t1.qtKill,
    t1.qtHs,
    t3.descMedal

from tb_lobby_stats_player as t1

left join tb_players_medalha as t2
on t1.idPlayer = t2.idPlayer

left join tb_medalha as t3
on t2.idMedal = t3.idMedal
*/

