import pandas as pd
import matplotlib.pyplot as plt

DATA_SOURCE = "./models/ds.xlsx"
STATES = ['CA', 'TX', 'NM', 'AZ']

CLEAN_MSN = 'RETCB'

TOTAL_MSN = 'TETCB'


def A(df_data):
    df_clean = pd.DataFrame(columns=['state', 'clean', 'total'])

    year = 2009
    for state in STATES:
        df_clean = df_clean.append({'clean': df_data[
            (df_data['MSN_data'] == CLEAN_MSN) & (df_data['year'] == year) & (df_data['state'] == state)][
            'data'].values[0],
                                    'total': df_data[(df_data['MSN_data'] == TOTAL_MSN) & (df_data['year'] == year) & (
                                            df_data['state'] == state)]['data'].values[0], 'state': state},
                                   ignore_index=True)
        df_clean['sp'] = df_clean['clean'] / df_clean['total'] * 100
    df_clean.plot(y='sp', x='state', kind='bar', legend=False, rot=0)
    plt.title('Renewable energy as a percentage of total energy in 2009', fontweight='bold')
    plt.ylabel("percent(%)")
    plt.xlabel("state")
    plt.show()


def B(df_data):
    df_clean = pd.DataFrame(columns=['year'] + STATES)

    for year in range(1960, 2010):
        result = {state: df_data[(df_data['year'] == year) & (df_data['state'] == state) & (
                df_data['MSN_data'] == CLEAN_MSN)]['data'].sum() / df_data[
                                               (df_data['year'] == year) & (df_data['state'] == state) & (
                                                       df_data['MSN_data'] == TOTAL_MSN)]['data'].values[0] * 100
                                     for state in STATES}
        result['year'] = year
        df_clean = df_clean.append(result, ignore_index=True)
    print(df_clean)
    # for state in STATES:
    #     for year in range(1960, 2010):
    #         df_clean = df_clean.append({'clean': df_data[
    #             (df_data['MSN_data'] == CLEAN_MSN) & (df_data['state'] == state)][
    #             'data'].values[0],
    #                                     'total':
    #                                         df_data[(df_data['MSN_data'] == TOTAL_MSN) & (df_data['year'] == year) & (
    #                                                 df_data['state'] == state)]['data'].values[0], 'state': state,
    #                                     'year': year},
    #                                    ignore_index=True)
    #         df_clean['sp'] = df_clean['clean'] / df_clean['total'] * 100
    # # print(df_clean)
    # #     df_clean['sp'] = df_clean['']
    # #     df_clean['sp'] = df_clean['clean'] / df_clean['total'] * 100
    # df_groups = df_clean.drop(['clean', 'total'], axis=1).groupby('state')
    # df_groups.plot(y='sp', x='year', rot=0)
    df_clean.plot(x='year')
    # df_clean.to_csv('./data/B-clean.csv')
    plt.title('Renewable energy as a percentage of total energy (1960 - 2009)', fontweight='bold')
    plt.ylabel("percent(%)")
    plt.xlabel("year")
    plt.show()


if __name__ == '__main__':
    # filename = './data/classification.csv'
    # df = pd.read_csv(filename) if os.path.exists(filename) else create_classification(filename)
    STATES = ['CA', 'TX', 'NM', 'AZ']
    df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])
    # A(df_data)
    B(df_data)
    # plt.savefig('./notebooks/1-2.png')
