import numpy as np
import pandas as pd

df = pd.read_csv('../csvs/rawWith2020.csv')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)
df.to_csv('../csvs/newWith2020.csv', index=False)