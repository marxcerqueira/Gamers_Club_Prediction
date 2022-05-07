drop table if exists tb_abt_sub;
create table tb_abt_sub as -- business problem
-- Qual a probabilidade de uma pessoa assinar a GC nos próximos 15 dias?
with tb_subs as (
    select t1.idPlayer,
        t1.idMedal,
        t1.dtCreatedAt,
        t1.dtExpiration,
        t1.dtRemove
    from tb_players_medalha as t1
        left join tb_medalha as t2 on t1.idMedal = t2.idMedal
    where t2.descMedal in ('Membro Premium', 'Membro Plus')
        and coalesce(t1.dtExpiration, date('now')) > t1.dtCreatedAt
)
select 
    t1.dtReff,
    t1.idPlayer,
    t1.qtPartidas,
    t1.qtPartidasMenor16,
    t1.qtDias,
    t1.qtDiasUltimaLobby,
    t1.mediaPartidaDias,
    t1.avgQtKill,
    t1.avgQtAssist,
    t1.avgQtDeath,
    t1.avgKDA,
    t1.KDAgeral,
    t1.avgKARound,
    t1.KARoundgeral,
    t1.avgQtHs,
    t1.txHsGeral,
    t1.avgHsRate,
    t1.avgQtBombeDefuse,
    t1.avgQtBombePlant,
    t1.avgQtTk,
    t1.avgQtTkAssist,
    t1.avgQt1Kill,
    t1.avgQt2Kill,
    t1.avgQt3Kill,
    t1.avgQt4Kill,
    t1.sumQt4Kill,
    t1.avgQt5Kill,
    t1.sumQt5Kill,
    t1.avgQtPlusKill,
    t1.avgQtFirstKill,
    t1.avgVlDamage,
    t1.avgDamageRound,
    t1.DamageRoundGeral,
    t1.avgQtHits,
    t1.avgQtShots,
    t1.avgQtLastAlive,
    t1.avgQtClutchWon,
    t1.avgQtRoundsPlayed,
    t1.avgVlLevel,
    t1.avgQtSurvived,
    t1.avgQtTrade,
    t1.avgQtFlashAssist,
    t1.avgQtHitHeadshot,
    t1.avgQtHitChest,
    t1.avgQtHitStomach,
    t1.avgQtHitLeftAtm,
    t1.avgQtHitRightArm,
    t1.avgQtHitLeftLeg,
    t1.avgQtHitRightLeg,
    t1.avgFlWinner,
    t1.qtMiragePartida,
    t1.qtMirageVitorias,
    t1.qtNukePartida,
    t1.qtNukeVitorias,
    t1.qtInfernoPartida,
    t1.qtInfernoVitorias,
    t1.qtVertigoPartida,
    t1.qtVertigoVitorias,
    t1.qtAncientPartida,
    t1.qtAncientVitorias,
    t1.qtDust2Partida,
    t1.qtDust2Vitorias,
    t1.qtTrainPartida,
    t1.qtTrainVitorias,
    t1.qtOverpassPartida,
    t1.qtOverpassVitorias,
    t1.vlLevelAtual,
    t1.qtMedalDistict,
    t1.qtMedalhaAdquiridas,
    t1.qtPremium,
    t1.qtPlus,
    t1.flFacebook,
    t1.flTwitter,
    t1.flTwitch,
    t1.descCountry,
    t1.vlIdadePlayer,
    t1.vlDiasCadastro,
    case
        when t2.idMedal is null then 0
        else 1
    end as flagSub

from tb_book_players as t1

    left join tb_subs as t2 on t1.idPlayer = t2.idPlayer -- pegando em 15 dias quem assinou e quem não assinou
    and t1.dtReff < t2.dtCreatedAt
    and t2.dtCreatedAt < date(t1.dtReff, '+15 day')
    
where AssinaturaAtiva = 0 --pegando apenas assinaturas novas (nenhum assinatura ativa)
and t1.dtReff < date('2022-02-01', '-15 day') --garantir maturação dos dados

;
