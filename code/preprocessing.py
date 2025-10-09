import numpy as np
import pandas as pd

if __name__ == '__main__':
    chunk_size = 100000
    chunks = []

    for chunk in pd.read_csv('../csvs/rawMoreTrimmed.csv', chunksize=chunk_size):
        chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
        chunk = chunk.dropna()
        chunks.append(chunk)

    df_cleaned = pd.concat(chunks, ignore_index=True)
    print(df_cleaned.shape)
    print(df_cleaned['Water'].value_counts())
    print(df_cleaned.head(15))
    df_cleaned.to_csv('cleanedMoreTrimmed.csv', index=False)