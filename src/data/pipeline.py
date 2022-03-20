import pandas as pd

class ImputeMissingValues():
    def __init__(self, cols, strategy):
        self.cols_ = cols
        self.strategy_ = strategy
    
    def transform(self, X, **params):
        X = X.copy()
        if self.strategy_ == 'mode':
            for col in self.cols_:
                fillval = X[col].mode().max()
                X[col] = X[col].fillna(fillval)

        if self.strategy_ == 'mean':
            for col in self.cols_:
                fillval = X[col].mean().max()
                X[col] = X[col].fillna(fillval)

        return(X)
    
    def fit (self, X, **params):
        return self