import pandas as pd
import shap
from sklearn.preprocessing import StandardScaler


class ShapModel:
    def __init__(self, model, period, y):

        self.model = model
        self.period = period
        self.colummns = y.columns
        

    def __call__(self, X):

        preds = self.model[1:].decision_function(X)
        preds = StandardScaler().fit_transform(preds)
        preds = pd.DataFrame(preds, columns=self.colummns)
        
        return preds[self.period]
    
masker = shap.maskers.Text(r'\W')