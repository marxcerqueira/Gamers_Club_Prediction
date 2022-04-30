# script em python para orquestrar a execução da query criada

# %%
#imports

import sqlalchemy
import datetime

from tqdm import tqdm

#%%

#Criando função date to list onde eu passo um intervalo de datas (data de incio e data de fim)
#e a função retorna uma list com todas as datas
def dates_to_list(dt_start, dt_stop):

    date_start = datetime.datetime.strptime(dt_start, '%Y-%m-%d')
    date_stop  = datetime.datetime.strptime(dt_stop, '%Y-%m-%d')
    days = (date_stop - date_start).days

    #loop para adicionar dias na data de inicio até chegar a data final
    dates = [(date_start + datetime.timedelta(i)).strftime('%Y-%m-%d') for i in range(days+1)]
    return dates

def backfill(query, con, dt_start, dt_stop):
    dates = dates_to_list(dt_start, dt_stop)

    for d in tqdm(dates):
        process_date(query, d, con)

# função que importa um arquivo de query
def import_query(path):
    with open(path, 'r') as open_file:
        query = open_file.read()

    return query

# função para processamento da query
def process_date(query ,date,con):
    #delete in case of same data inputation
    delete = f"delete from tb_book_players where dtReff = '{date}'"
    con.execute(delete)

    query = query.format(date = date)
    con.execute(query)
# %%

# conexão com o banco de dados sqlite
con = sqlalchemy.create_engine('sqlite:///../data/gc.db')

query = import_query('query.sql')

dt_start = input('Entre com uma data de início:')
dt_stop = input('Entre com uma data de fim:')

backfill(query, con, dt_start, dt_stop)
