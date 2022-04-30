# script em python para orquestrar a execução da query criada

# %%
#imports

import sqlalchemy
# %%

# função que importa um arquivo de query
def import_query(path):
    with open(path, 'r') as open_file:
        query = open_file.read()

    return query

# função para processamento da query
def process_date(query ,date,engine):
    #delete in case of same data inputation
    delete = f"delete from tb_book_players where dtReff = '{date}'"
    engine.execute(delete)

    query = query.format(date = date)
    engine.execute(query)
# %%

# conexão com o banco de dados sqlite
engine = sqlalchemy.create_engine('sqlite:///../data/gc.db')

query = import_query('query.sql')

date = '2022-01-01'

process_date(query ,date,engine)
