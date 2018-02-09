import pandas as pd
import os
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

DATA_SOURCE = "./models/ds.xlsx"
FACTOR_NUMBER = 2


def create_purify(year: int, state: str):
    def add_suffix_b_names(x):
        x['MSN_data'] = x['MSN'].str.slice(0, 4) + 'B'
        return x

    df_msn = pd.read_excel(DATA_SOURCE, "msncodes", names=['MSN', 'desc', 'unit'])
    df_data = pd.read_excel(DATA_SOURCE, "seseds", names=['MSN_data', 'state', 'year', 'data'])

    # Specify a year and state
    df_data_2005_CA = df_data[(df_data['state'] == state) & (df_data['year'] == year)]

    # Get suffix names of the MSN node are end with 'P'
    df_msn_suffix_p = df_msn[df_msn['MSN'].str.contains('P$')]

    # Using groupBy to purify the group
    df_purify = df_msn_suffix_p.groupby('unit').filter(lambda x: len(x) > 1)
    df_msn_group = df_purify.groupby('unit').apply(add_suffix_b_names)

    # Get the handled dataframe
    df_data_2005_CA_plus = df_msn_group.set_index('MSN_data').join(df_data_2005_CA.set_index('MSN_data'), how="inner")
    df_data_2005_CA_plus.to_csv('./purify.csv')
    return df_data_2005_CA_plus


def get_corr(df1, df2) -> float:
    cov = df1['data'].cov(df2['data'])
    std1 = df1['data'].std()
    std2 = df2['data'].std()
    return cov / (std1 * std2)


def get_loading_matrix(groups: list):
    corrMatrix = [([1] * len(groups)) for i in range(len(groups))]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_i = groups[i]['group'].reset_index()
            group_j = groups[j]['group'].reset_index()
            corrMatrix[i][j] = get_corr(group_i, group_j)
            corrMatrix[j][i] = corrMatrix[i][j]

    # np.linalg.eig return a tuple, 0 is eigenvalues
    eigvalMatrix = pd.DataFrame(np.linalg.eig(corrMatrix)[0], columns=['value'])
    eigvalMatrix['name'] = [group['name'] for group in groups]
    factor_eigval_matrix = eigvalMatrix.nlargest(FACTOR_NUMBER, 'value')
    factor_eigval_matrix['percent'] = pd.DataFrame(factor_eigval_matrix['value'] / eigvalMatrix['value'].sum() * 100)
    eigvecMatrix = pd.DataFrame(np.linalg.eig(corrMatrix)[1])

    loading_matrix = eigvecMatrix * np.sqrt(factor_eigval_matrix['value'])
    loading_matrix['name'] = [group['name'] for group in groups]
    return loading_matrix, factor_eigval_matrix


def get_weights(loading_matrix: pd.DataFrame, factor_eigval_matrix):
    df = loading_matrix[loading_matrix.columns.difference(['name'])].abs() * factor_eigval_matrix['percent']
    df['name'] = loading_matrix['name']
    df['sum'] = df.sum(axis=1)
    df['weight'] = df['sum'] / df['sum'].sum() * 100
    return df


def paint():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph
    cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    ax.set_xlim(-40, 40)
    ax.set_ylim(-40, 40)
    ax.set_zlim(-100, 100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('./purify.csv') if os.path.exists('./purify.csv') else create_purify(2005, 'CA')
    unit_list = list(df['unit'].unique())
    groups = [{"name": name, "group": group} for name, group in df.groupby(['unit'])]
    loading_matrix, factor_eigval_matrix = get_loading_matrix(groups)
    df_weight = get_weights(loading_matrix, factor_eigval_matrix)

    df_final = df_weight[['sum', 'name', 'weight']]
    paint()