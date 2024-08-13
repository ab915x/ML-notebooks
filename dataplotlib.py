import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def make_heatmap(feature_cols, df, target, figure_size=(12,7), dropnans=True):
    '''Функция для отрисовки тепловой карты, безразличная к типу данных'''
    df_func = df.copy()
    if dropnans:
        df_func.dropna(inplace=True)
    cat_cols = df_func.select_dtypes('object').columns
    bool_cols = df_func.select_dtypes('bool').columns
    df_func[bool_cols] = df_func[bool_cols].astype(int)
    for column in cat_cols:
        dictionary = {}
        for index, value in enumerate(df_func[column].unique()):
            dictionary[value] = index
        df_func[column] = df_func[column].map(dictionary)
    heatmapdf = df_func[feature_cols].copy()
    if isinstance(target, str):
        col = df_func[target]
        heatmapdf.drop(axis=1, columns=[target], inplace=True)
    else:
        col = target
        heatmapdf.drop(axis=1, columns=[col.name], inplace=True)
    heatmapdf.insert(0, col.name, col)
    fig = plt.figure(figsize=figure_size)
    num_matrix = heatmapdf.corr(method='spearman')
    sns.heatmap(num_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Heatmap')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.show()
