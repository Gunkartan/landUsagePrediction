import pandas as pd

df = pd.read_csv('../csvs/preprocessed.csv')
print(list(df.columns))
print(df['Unnamed: 0'])
print(df.head(15))