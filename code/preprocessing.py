import pandas as pd

df = pd.read_csv('../csvs/kaimuk.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df.to_csv('../csvs/preprocessed.csv', index=False)