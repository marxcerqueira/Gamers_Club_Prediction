-- Pegando as primeiras 3 partidas de cada player da base
with tb_lobbies as (

    select
        *,
        row_number() over(partition by idPlayer order by idLobbyGame) AS lineNumber
    from tb_lobby_stats_player
    
    order by idPlayer
)

select * from tb_lobbies
where lineNumber <= 3

