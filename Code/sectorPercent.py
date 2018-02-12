import pandas as pd
import matplotlib.pyplot as plt
from cleanPart import create_clean_sector

DATA_SOURCE = "./models/ds.xlsx"
STATES = ['CA', 'TX', 'NM', 'AZ']

SECTOR_MSN = {
    'transportation': ['EMACB'],
    'commercial': ["EMCCB", "GECCB", "HYCCB", "SOCCB", "WWCCB", "WYCCB"],
    'electric power': ["HYEGB", "GEEGB", "SOEGB", "WWEIB", "WYEGB"],
    'industrial': ["EMICB", "EMLCB", "GEICB", "HYICB", "SOICB", "WWICB", "WYICB"],
    'residential': ["WDRCB", "GERCB", "SORCB"]
}

if __name__ == '__main__':
    years = range(1960, 2010)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
    df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])
    df_total_years = []
    for year in years:
        df = create_clean_sector(year, df_data, True)
        df['year'] = year
        df_total_years.append(df)
    df_total_years = pd.concat(df_total_years)
    df_total_years = df_total_years[[sector + '_sp' for sector in SECTOR_MSN] + ['year', 'state']]
    df_total_years = df_total_years.rename(columns={sector + '_sp': sector for sector in SECTOR_MSN})

    df_groups = df_total_years.groupby('state')
    for (state, group), ax in zip(df_groups, axes.flatten()):
        group.plot(x='year', ax=ax, title="The proportion of different sector of {} state in renewable energy usage".format(state), kind='area')
        plt.ylabel("percent(%)")
    plt.xlabel("year")
    fig.tight_layout()
    plt.savefig('./notebooks/B-percent.png')
    # plt.show()
    plt.show()
