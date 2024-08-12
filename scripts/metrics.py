import pandas as pd
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
            
            return pd.Series(result)
        
    y_pred = pd.DataFrame(y_pred, index=y_true.index, columns=y_true.columns)
    df = pd.concat([y_true, y_pred], keys=['y_true', 'y_pred'])
    results = df.apply(f, scorer=scorer, result_type='expand', **kwargs)
    return results.T

def get_scores(y_true, y_pred, threshold=0):

    acc = multioutput_scorer(balanced_accuracy_score, y_true>0, y_pred>threshold)
    acc = acc.unstack().mean()['Score'].rename('acc')
    pearson = multioutput_scorer(pearsonr, y_true, y_pred)
    pearson = pearson.unstack().mean()['Score'].rename('pearson')
    scaler = StandardScaler()
    t = pd.DataFrame(scaler.fit_transform(y_true), y_true.index, y_true.columns)
    p = pd.DataFrame(scaler.transform(y_pred), y_true.index, y_true.columns)
    mse = multioutput_scorer(mean_squared_error, t, p, squared=True)
    mse = mse.unstack().mean()['Score'].rename('mse')
    
    return pd.concat([acc, mse, pearson], axis=1)