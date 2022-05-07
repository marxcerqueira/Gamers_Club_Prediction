# %%
import pandas as pd
import sqlalchemy

# %%
# sample
conn = sqlalchemy.create_engine("sqlite:///../../../data/gc.db")

df = pd.read_sql_table('tb_abt_sub', conn)

df.head()

# %%
