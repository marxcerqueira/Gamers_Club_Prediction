-- saber em média quantas medalhas distintas um player tem
/*

select 
    avg(distMedal) as avgDistMedal,
    min(distMedal) as minQtyMedal,
    max(distMedal) as maxQtyMedal

from (
select
        idPlayer,
        count(distinct idMedal) as distMedal
    
    from tb_players_medalha
    where flActive = 1
    
    group by idPlayer
    
    order by distMedal desc
    )

    */


-- saber a tx de hs da Tribo
/* SELECT
    flTribo,
    avg(txHs) as avgTxHs

from(
    select
        t1.idPlayer,
        t1.qtKill,
        t1.qtHs,
        round(100.0 * t1.qtHs/t1.qtKill, 2) as txHs,
        coalesce(t2.flTribo, 0) as flTribo


    from tb_lobby_stats_player as t1

    left join (

        select
            t1.idPlayer,
            max(1) as flTribo

        from tb_players_medalha as t1

        where t1.idMedal in (8, 37)
        and t1.flActive = 1

        group by t1.idPlayer

    ) as t2
    on t1.idPlayer = t2.idPlayer
    )

    group by flTribo */

-- solving the same problem with clause WITH, better way to use
-- subquery

/* with 
tb_tribo as (

    select
        t1.idPlayer,
        max(1) as flTribo

        from tb_players_medalha as t1

        where t1.idMedal in (8, 37)
        and t1.flActive = 1

        group by t1.idPlayer
),

tb_hs as (
    select
        t1.idPlayer,
        round(avg(100.0 * t1.qtHs/t1.qtKill), 2) as AvgTxHs
    
    from tb_lobby_stats_player as t1
    group by t1.idPlayer
)
,
tb_tribo_hs_join as(
    select
        t1.*,
        coalesce(t2.flTribo, 0) as flTribo

    from tb_hs as t1

    left join tb_tribo as t2
    on t1.idPlayer = t2.idPlayer
)

select
    flTribo,
    avg(AvgTxHs) as txHsTribo

from tb_tribo_hs_join
group by flTribo */

-- qual winrate dos assinantes Plus vs Premium

with 
tb_premium_plus_players as (
    select
        distinct t1.idPlayer,
        case when t1.idMedal = 1 then 'premium' else 'plus' end SubType

    from tb_players_medalha as t1

    where t1.idMedal in (1, 3)
    and t1.flActive = 1
),

tb_win_rate as (
    select
        t2.idPlayer,
        round(avg(t2.flWinner),2) as winRate

    from tb_lobby_stats_player as t2

    group by 1
),

tb_subs_stats as (
    select 
        t1.*,
        coalesce(t2.SubType, 'não sub') as SubType

    from tb_win_rate as t1

    left join tb_premium_plus_players as t2
    on t1.idPlayer = t2.idPlayer
)


select 
    SubType,
    round(avg(winRate),4) as avgWinRate

from tb_subs_stats

group by SubType

order by avgWinRate desc
