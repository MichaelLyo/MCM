import pandas as pd
import os
from yaml import load
import matplotlib.pyplot as plt

from profile_1 import create_purify

DATA_SOURCE = "./models/ds.xlsx"

MSN_SOURCE = load(open('./models/msn_source.yaml', 'r', encoding='utf-8'))
MSN_SECTOR = load(open('./models/msn_sector.yaml', 'r', encoding='utf-8'))
EXCLUDE_MSN = ["ROPRB", "TEACB", "TECCB", "TEEIB", "TEICB", "TEPFB", "TEPRB", "TERCB", "TERFB", "TETCB", "TETPB",
               "TETXB", "TNACB", "TNCCB", "TNICB", "TNRCB", "TNSCB", "TNTXB"]

RENAMES = {
    'Thousand barrels': 'petroleum products',
    'Thousand short tons': 'coal',
    'Million kilowatthours': 'Electricity',
    'Million cubic feet': 'natural gas'
}


def add_suffix_b_names(x):
    return MSN_SECTOR[x['MSN_data'][2:4]]


def paint(df, state):
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(16, 12))
    # df_groups = df.groupby(['unit', 'sector'])
    units = df['unit'].unique()
    sectors = df['sector'].unique()
    r = -1
    c = -1
    years = df['year'].unique()
    for unit in units:
        r = r + 1
        c = -1
        for sector in sectors:
            c = c + 1
            df1 = df[(df['unit'] == unit) & (df['sector'] == sector)]
            if (len(df1) > 0):
                df1[['year', 'data']].set_index('year').plot(xticks=range(years[0], years[-1], 10),
                                                             title='{}-{}'.format(sector, RENAMES[unit]), ax=axes[r, c],
                                                             legend=False)
            else:
                df_empty = pd.DataFrame()
                df_empty['year'] = df['year']
                df_empty['data'] = 0
                df_empty.set_index('year').plot(xticks=range(years[0], years[-1], 10),
                                                title='{}-{}'.format(sector, RENAMES[unit]), ax=axes[r, c],
                                                legend=False)
                # plt.plot(title=, ax=, legend=False)
    # for (info, group), ax in zip(df_groups, axes.flatten()):
    #     product = RENAMES[info[0]]
    #     sector = info[1] + ' sector'Z
    #     group.plot(x='year', ax=ax, title="{}-{}".format(sector, product), legend=False)
    plt.ylabel("Billion Btu")
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle("Evolution of energy profile of {} (1960 - 2009)".format(state))
    plt.savefig('./notebooks/B-classify-{}.png'.format(state))


def create_classification(df):
    df_classification = df[df['MSN_data'].str.slice(2, 4).isin(MSN_SECTOR)]
    df_classification = df_classification.reset_index(drop=True)
    df_classification['sector'] = df_classification.apply(add_suffix_b_names, axis=1)
    sectors = list(df_classification['sector'].unique())
    groups = df_classification.groupby(['year', 'sector', 'unit'])['data'].sum()
    groups = groups.to_frame()

    return groups


def load_df(years, state):
    dfs = []
    for year in years:
        filename = './data/{}-{}.csv'.format(year, state)
        dfs.append(pd.read_csv(filename) if os.path.exists(filename) else create_purify(year, state, filename))
    return pd.concat(dfs)


if __name__ == '__main__':
    filename = './data/classification.csv'
    # df = pd.read_csv(filename) if os.path.exists(filename) else create_classification(filename)
    STATES = ['CA', 'TX', 'NM', 'AZ']
    years = range(1960, 2010)
    for state in STATES:
        filename = './data/B-classify-{}.csv'.format(state)
        df = load_df(years, state)
        df_classification = create_classification(df)
        df_classification.to_csv(filename)
        df = pd.read_csv(filename, names=['year', 'sector', 'unit', 'data'], header=0)
        paint(df, state)
