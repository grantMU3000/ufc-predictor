'''
Using duckdb for analyzing the raw Greco1899 scraped UFC stats & results.

'''

import duckdb

con = duckdb.connect()  # in-memory, no file/account needed

df = con.execute(
    "SELECT * FROM read_csv_auto('data/raw/ufc_event_details.csv')"
).df()

print(df.shape)
print(df.dtypes)
print(df.describe())
print(df.isnull().mean())  # null rate per column
print(df.head())