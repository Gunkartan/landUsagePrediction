import numpy as np
import pandas as pd

df = pd.read_csv('../csvs/rawWithEvenMoreSamples.csv')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)
df.to_csv('../csvs/newWithEvenMoreSamples.csv', index=False)