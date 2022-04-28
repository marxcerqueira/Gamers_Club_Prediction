SELECT
    idPlayer,
    descCountry,
    case when descCountry = 'br' THEN 'huehue'
         when descCountry IN ('ar', 'pe', 'uy', 'py', 'cl') then 'manito'
         when descCountry IN ('ca', 'us') then 'NA'
         else 'foda-se'
    end as nacionalidade

from tb_players