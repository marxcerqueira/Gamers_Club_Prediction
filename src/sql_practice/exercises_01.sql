/* exercícios GC
1) Selecione todos os players argentinos*/

/*select
    idPlayer,
    descCountry,
    dtBirth
from tb_players
WHERE descCountry = 'ar'*/ 

/*
select
    count(*) as qty_ar_players,
    descCountry
from tb_players
where descCountry = 'ar';*/

/*2) Selecione todos os players brasileiros que nasceram em 92*/

/*

select
    idPlayer,
    descCountry,
    dtBirth,
    strftime('%Y', dtBirth) as yrBirth
from tb_players
where strftime('%Y',dtBirth) = '1992'
and descCountry = 'br';

*/

/*3) Selecione players com medalhas ativas*/

--select * from tb_players_medalha;

/*

select
    tb_players_medalha.idPlayer,
    tb_players_medalha.idMedal,
    tb_medalha.descMedal
from tb_medalha
inner join tb_players_medalha on tb_medalha.idMedal = tb_players_medalha.idMedal 
where tb_players_medalha.flActive = 1;

*/
/*

select
    count(*) as playersActiveMedal
from tb_players_medalha
where tb_players_medalha.flActive = 1;
*/

-- 4) Selecione players que tiveram mais de 50% de hs em uma partida
-- txHs = qtHs/qt1Kill

select
      idLobbyGame,
      idPlayer,
      qtKill,
      qtHs,
      qtDeath,
      round(1.0*qtHs/qtKill,2) as txHS,
      round(1.0*qtHs/qtKill,2)*100 as txHSperc
from tb_lobby_stats_player
where txHSperc  > 50;