from supplychainpy.model_decision import analytical_hierarchy_process
from cleanPart import create_clean_sector
import pandas as pd
from math import exp
import os
import matplotlib.pyplot as plt

DATA_SOURCE = "./models/ds.xlsx"
STATES = ['CA', 'TX', 'NM', 'AZ']

SECTOR_MSN = {
    'transportation': ['EMACB'],
    'commercial': ["EMCCB", "GECCB", "HYCCB", "SOCCB", "WWCCB", "WYCCB"],
    'electric power': ["HYEGB", "GEEGB", "SOEGB", "WWEIB", "WYEGB"],
    'industrial': ["EMICB", "EMLCB", "GEICB", "HYICB", "SOICB", "WWICB", "WYICB"],
    'residential': ["WDRCB", "GERCB", "SORCB"]
}


def judge(a, b):
    x = [a, b] if a > b else [b, a]
    if x[1] == 0:
        x[1] = 0.000000001
    c = x[0] / x[1]
    p = 0
    if 1 < c <= 2.5:
        p = 2
    elif 2.5 < c <= 4.5:
        p = 3
    elif 4.5 < c <= 6:
        p = 4
    elif 6 < c <= 8:
        p = 5
    elif 8 < c <= 9.5:
        p = 6
    elif 9.5 < c <= 11.5:
        p = 7
    elif 11.5 < c <= 13:
        p = 8
    elif c == 1:
        p = 1
    else:
        p = 9
    return p if a > b else 1 / p


def get_matrix(l):
    matrix = [([1] * len(l)) for i in range(len(l))]
    for i in range(len(l)):
        for j in range(i, len(l)):
            matrix[i][j] = judge(l[i], l[j])
            matrix[j][i] = 1 / matrix[i][j]
    return matrix


def create_scores(years, filename, df_data, df_people=None):
    sectors = [msn for msn in SECTOR_MSN]
    df_score = pd.DataFrame(columns=['year'] + STATES)
    for year in years:
        df = create_clean_sector(year, df_data, True)
        clean_sectors_l = list(df[sectors].sum() / df['total'].sum() * 100)
        clean_sectors_matrix = get_matrix(clean_sectors_l)
        states_sectors_matrix = {}


        for sector in sectors:
            if df_people is not None:
                df = df[sectors].divide(df_people[str(year)], axis=0)
            sp = list(df[sector])
            states_sectors_matrix[sector] = get_matrix(sp)

        # lorry_cost = {'CA': 50000, 'TX': 25000, 'NM': 5000, 'AZ': 15000}
        criteria = sectors
        criteria_scores = clean_sectors_matrix

        options = STATES
        option_scores = states_sectors_matrix
        lorry_decision = analytical_hierarchy_process(criteria=tuple(criteria),
                                                      criteria_scores=criteria_scores,
                                                      options=tuple(options),
                                                      option_scores=option_scores,
                                                      quantitative_criteria=())
        # res = lorry_decision['cost_benefit_ratios']
        lorry_decision['year'] = year
        df_score = df_score.append(lorry_decision, ignore_index=True)
    df_score['year'] = df_score['year'].astype(int)
    df_score[STATES] = df_score[STATES] * 100
    df_score.to_csv(filename)
    return df_score


def predict(year):
    result = {
        'NM': 0.05263 * year - 104.64807,
        'AZ': exp(-0.10513 * year + 213.24602),
        'CA': exp(-0.34649 * year + 697.98712) + 50
    }
    result['TX'] = 100 - result['NM'] - result['AZ'] - result['CA']
    result['year'] = year
    return result


def total():
    fig = plt.figure(figsize=(15, 15))
    open_prediction = True
    filename = './data/B-total-level.csv'

    years = range(1995, 2010)
    future_years = range(2010, 2051)
    if not os.path.exists(filename):
        df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])
        df_score = create_scores(years, filename, df_data)
    else:
        df_score = pd.read_csv(filename, header=0, names=['year'] + STATES)

        # for state in STATES:
    pic_path = "./notebooks/B-level.png"
    title = "Interstate difference of AHP result (1960 - 2009)"
    if open_prediction:
        for year in future_years:
            df_score = df_score.append(predict(year), ignore_index=True)

    df_score.plot(x='year', kind='line')

    if open_prediction:
        pic_path = "./notebooks/D-predict.png"
        title = 'Prediction based on regression results (2025 - 2050)'
        plt.axvspan(2025, 2050, facecolor='#2ca02c', alpha=0.2)
        plt.annotate('2025', xy=(2025, 70), xycoords='data',
                     xytext=(0.8, 0.95), textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top',
                     )
        plt.axvline(x=2025, color="#333333", linewidth=1)

    plt.title(title, fontweight='bold')
    plt.ylabel("AHP result(%)")
    plt.xlabel("year")
    # plt.show()
    plt.savefig(pic_path)
    plt.show()

def person():
    fig = plt.figure(figsize=(15, 15))
    open_prediction = False
    df_people = pd.read_csv('./models/people-min.csv')
    filename = './data/B-total-level-percapita.csv'
    years = range(1960, 2010)
    future_years = range(2010, 2051)
    if not os.path.exists(filename):
        df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])
        df_score = create_scores(years, filename, df_data, df_people)
    else:
        df_score = pd.read_csv(filename, header=0, names=['year'] + STATES)

        # for state in STATES:
    pic_path = "./notebooks/B-level-percapita.png"
    title = "Interstate difference of AHP result per capita (1960 - 2009)"
    if open_prediction:
        for year in future_years:
            df_score = df_score.append(predict(year), ignore_index=True)

    df_score.plot(x='year', kind='line')

    if open_prediction:
        pic_path = "./notebooks/D-predict-percapita.png"
        title = 'Prediction based on regression results (2025 - 2050)'
        plt.axvspan(2025, 2050, facecolor='#2ca02c', alpha=0.2)
        plt.annotate('2025', xy=(2025, 70), xycoords='data',
                     xytext=(0.8, 0.95), textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     horizontalalignment='right', verticalalignment='top',
                     )
        plt.axvline(x=2025, color="#333333", linewidth=1)

    plt.title(title, fontweight='bold')
    plt.ylabel("AHP result(%)")
    plt.xlabel("year")
    # plt.show()
    plt.savefig(pic_path)
    plt.show()

if __name__ == '__main__':
    # person(
    total()
