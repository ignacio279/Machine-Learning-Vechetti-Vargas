from sklearn.base import BaseEstimator, TransformerMixin

class IncrementalFreqEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols, alpha=1.0, bucket_other=True):
        self.cols         = cols
        self.alpha        = alpha
        self.bucket_other = bucket_other

    def fit(self, X, y=None):
        self.N_      = len(X)
        self.counts_ = {col: X[col].value_counts().to_dict()
                        for col in self.cols}
        self.K_ = {col: len(self.counts_[col]) for col in self.cols}
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.cols:
            cnts = self.counts_[col]
            N    = self.N_
            K    = self.K_[col]
            α    = self.alpha

            freq_map = {cat: (cnt + α) / (N + α * K)
                        for cat, cnt in cnts.items()}
            default = α / (N + α * K)

            if self.bucket_other:
                known = set(cnts.keys())
                X[col] = X[col].where(X[col].isin(known), 'Other')
                freq_map['Other'] = default

            X[col + '_freq'] = X[col].map(freq_map).fillna(default)

        return X

    def partial_fit(self, X, y=None):
        """
        (Opcional) incorporar X a tu histórico:
        suma conteos y actualiza N_, K_, counts_
        """
        for col in self.cols:
            new_counts = X[col].value_counts().to_dict()
            for cat, cnt in new_counts.items():
                self.counts_[col][cat] = self.counts_[col].get(cat, 0) + cnt
            self.K_[col] = len(self.counts_[col])
        self.N_ += len(X)
        return self