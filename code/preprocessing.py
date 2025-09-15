import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../csvs/rawWithMNDWI.csv')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    dfCleaned = df.dropna().reset_index(drop=True)
    dfCleaned.to_csv('cleaned.csv', index=False)