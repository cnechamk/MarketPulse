import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

def multioutput_scorer(scorer, y_true, y_pred, **kwargs):
    def f(series, scorer, **kwargs):
        score = scorer(series['y_true'], series['y_pred'])
        try:
            ci = score.confidence_interval()
        except AttributeError:
            return {'Score':score}
        else:
            result = {'Score':score[0]}
            result.update(zip(['low', 'high'],ci))
            # print(result)
            return pd.Series(result)
    y_pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
    df = pd.concat([y_true, y_pred], keys=['y_true', 'y_pred'])
    results = df.apply(f, scorer=scorer, result_type='expand', **kwargs)
    return results.T

def get_scores(y_true, y_pred, thresholds):

    acc = multioutput_scorer(balanced_accuracy_score, y_true>0, y_pred>thresholds)
    acc = acc.unstack().mean()['Score'].rename('acc')
    pearson = multioutput_scorer(pearsonr, y_true, y_pred)
    pearson = pearson.unstack().mean()['Score'].rename('pearson')
    scaler = StandardScaler()
    t = pd.DataFrame(scaler.fit_transform(y_true), y_true.index, y_true.columns)
    p = pd.DataFrame(scaler.transform(y_pred), y_true.index, y_true.columns)
    mse = multioutput_scorer(mean_squared_error, t, p, squared=True)
    mse = mse.unstack().mean()['Score'].rename('mse')
    
    return pd.concat([acc, mse, pearson], axis=1)

def display_graphs(y_true, y_pred, threshold=0):
    results = multioutput_scorer(balanced_accuracy_score, y_true>threshold, y_pred>threshold)
    results = results.reset_index()
    grid = sns.relplot(results, y='Score', x='Period', hue='Ticker', kind='line', col='Ticker', col_wrap=3);
    for ticker, grp in results.groupby('Ticker'):
        ax=grid.axes_dict[ticker]
        ax.hlines(.5, grp['Period'].min(), grp['Period'].max(), color='black')
    grid.fig.suptitle('Balanced Accuracy Score');
    grid.fig.set_constrained_layout(True)

    # %%
    results = multioutput_scorer(pearsonr, y_true, y_pred)
    results = results.reset_index()
    grid=sns.relplot(results, y='Score', x='Period', hue='Ticker', kind='line', col='Ticker', col_wrap=3);
    for ticker, grp in results.groupby('Ticker'):
        ax=grid.axes_dict[ticker]
        ax.fill_between('Period', 'low', 'high', data=grp, color='lightgrey')
        ax.hlines(0, grp['Period'].min(), grp['Period'].max(), color='black')
    grid.fig.suptitle('Spearman Correlation with Confidence Intervals');
    grid.fig.set_constrained_layout(True)

    # %%
    scaler = StandardScaler()
    t = pd.DataFrame(scaler.fit_transform(y_true), y_true.index, y_true.columns)
    p = pd.DataFrame(scaler.transform(y_pred), y_true.index, y_true.columns)
    results = multioutput_scorer(mean_squared_error, t, p, squared=False)
    results = results.reset_index()
    grid = sns.relplot(results, y='Score', x='Period', hue='Ticker', kind='line', col='Ticker', col_wrap=3);
    for ticker, grp in results.groupby('Ticker'):
        ax=grid.axes_dict[ticker]
        ax.set_ylim(0,2)
        ax.hlines(1, grp['Period'].min(), grp['Period'].max(), color='black')
    grid.fig.suptitle('Root Mean Squared Error (scaled - a score of 1 is no better than choosing the mean)');
    grid.fig.set_constrained_layout(True)