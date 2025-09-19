import numpy as np
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../csvs/rawWithStatistics.csv')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(df.shape)
    print(df['Water'].value_counts())
    print(df.head(15))
    dfCleaned = df.dropna().reset_index(drop=True)
    print(dfCleaned.shape)
    print(dfCleaned['Water'].value_counts())
    print(dfCleaned.head(15))
    dfCleaned.to_csv('cleaned.csv', index=False)