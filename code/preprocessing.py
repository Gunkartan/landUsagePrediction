import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../csvs/allYears.csv')
    dfCleaned = df.dropna()
    dfCleaned.to_csv('cleaned.csv', index=False)