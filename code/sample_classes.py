import pandas as pd

def stratified_sample(group: pd.DataFrame):
    if group.Name == 13:
        n = min(len(group), n_samples_other)

    else:
        n = min(len(group), n_samples_crop)

    return group.sample(n, random_state=42)

if __name__ == '__main__':
    df = pd.read_csv('../csvs/rawCrop.csv')
    n_samples_crop = 3000
    n_samples_other = 10000
    df_sampled = df.groupby('Crops', group_keys=False).apply(stratified_sample)
    df_sampled.to_csv('../csvs/rawCrop.csv', index=False)