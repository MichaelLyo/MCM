import pandas as pd
import os
from yaml import load
import matplotlib.pyplot as plt

from classify import create_classification

DATA_SOURCE = "./models/ds.xlsx"

CLEAN_MSN = 'RETCB'

TOTAL_MSN = 'TETCB'

SECTOR_MSN = {
    'transportation': ['EMACB'],
    'commercial': ["EMCCB", "GECCB", "HYCCB", "SOCCB", "WWCCB", "WYCCB"],
    'electric power': ["HYEGB", "GEEGB", "SOEGB", "WWEIB", "WYEGB"],
    'industrial': ["EMICB", "EMLCB", "GEICB", "HYICB", "SOICB", "WWICB", "WYICB"],
    'residential': ["WDRCB", "GERCB", "SORCB"]
}

def create_clean_sector(year, df_data, percent: bool):
    STATES = ['CA', 'TX', 'NM', 'AZ']
    # df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])
    df = pd.DataFrame(columns=[sector for sector in SECTOR_MSN] + ['state', 'total'])
    for state in STATES:
        df_purify = df_data[(df_data['state'] == state) & (df_data['year'] == year)]
        total_val = df_purify[df_purify['MSN_data'] == CLEAN_MSN]['data'].values[0]
        sums = {'state': state, 'total': total_val}
        for sector in SECTOR_MSN:
            msns = SECTOR_MSN[sector]
            if percent:
                sums[sector + '_sp'] = df_purify[df_purify['MSN_data'].isin(msns)]['data'].sum() / total_val * 100
            sums[sector] = df_purify[df_purify['MSN_data'].isin(msns)]['data'].sum()
        df = df.append(sums, ignore_index=True)
    return df

if __name__ == '__main__':
    filename = './data/classification.csv'
    df = pd.read_csv(filename) if os.path.exists(filename) else create_classification(filename)

    fig = plt.figure()
    df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])
    year = 2009

    df.plot(x='state', kind='bar', rot=0)
    plt.title('The proportion of different sector in renewable energy usage in 2009', fontweight='bold')
    plt.ylabel("percent(%)")
    plt.xlabel("state")
    plt.savefig('./notebooks/1-3.png')
    plt.show()