import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_stack_ys_line(y_pred_p: str, y_true_p: str):
    """
    Save FacetGrid, each row of grid is ticker, each plot on grid x: date, y: y, hue: y_pred | y_true

    Args:
        y_pred: path to dataframe as saved in repro_sklearn2.py
        y_true: path to dataframe as saved in repro_sklearn2.py
    """
    def melt_y(y: pd.DataFrame):
        df = pd.melt(y, id_vars='date')
        return df

    y_pred = pd.read_csv(y_pred_p, header=[0, 1], index_col=0)
    y_true = pd.read_csv(y_true_p, header=[0, 1], index_col=0)

    first_level_cols = y_pred.columns.get_level_values(0).unique()


    for l1_col in first_level_cols:
        df_pred = melt_y(y_pred.loc[:, l1_col].reset_index(names=['date']))
        df_true = melt_y(y_true.loc[:, l1_col].reset_index(names=['date']))

        df_pred['type'] = 'pred'
        df_true['type'] = 'true'

        df = pd.concat((df_pred, df_true))
        df.date = pd.to_datetime(df.date)

        g = sns.FacetGrid(df, row="Ticker", hue='type', sharex=True, sharey=False, height=6, aspect=2)
        g.map_dataframe(sns.lineplot, x='date', y='value')
        plt.title(f"Period: {l1_col}")
        plt.tight_layout()
        plt.savefig(f'data/viz/period_{l1_col}.png')
        plt.close()


if __name__ == "__main__":
    y_pred_p = "data/tmp/y_pred.csv"
    y_true_p = "data/tmp/y_true.csv"
    plot_stack_ys_line(y_pred_p=y_pred_p, y_true_p=y_true_p)
