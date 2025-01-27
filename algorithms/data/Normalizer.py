import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, StandardScaler


'''
This normalizer is specifically for the Glaucoma simulated feature set. The first three normalizations should theoretically work on any
fully numeric data set (no categorical data), however the fourth normalizer requires the feature R5 to be present.
'''
class Normalizer:
    def __init__(self, selection):
        self.which = selection
        self.norm_name = None
        self.s = None
        self.s2 = None

        self.data = None
        self.centers = None

    def norm(self, data, centers=None):
        if self.which == 0:    # Min Max Normalization
            self.norm_name = 'min_max_'
            self.s = MinMaxScaler()
            data = self.s.fit_transform(data)

            if centers is not None:
                centers = self.s.transform(centers)

        elif self.which == 1:  # Robust Scaler Normalization
            self.norm_name = 'robust_'
            self.s = RobustScaler()
            self.s2 = MinMaxScaler()

            data = self.s.fit_transform(data)
            data = self.s2.fit_transform(data)

            if centers is not None:
                centers = self.s.transform(centers)
                centers = self.s2.transform(centers)
            
        elif self.which == 2:  # Quantile Transform
            self.norm_name = 'quantile_'
            self.s = QuantileTransformer(output_distribution='normal')
            self.s2 = MinMaxScaler()

            data = self.s.fit_transform(data)
            data = self.s2.fit_transform(data)

            if centers is not None:
                centers = self.s.transform(centers)
                centers = self.s2.transform(centers)

        elif self.which == 3:  # Hyperbolic Tangent
            self.norm_name = 'tanh_'
            self.s = StandardScaler()
            self.s2 = MinMaxScaler()
            
            data = self.s.fit_transform(data)
            data = np.tanh(data)
            data = self.s2.fit_transform(data)

            if centers is not None:
                centers = self.s.transform(centers)
                centers = np.tanh(centers)
                centers = self.s2.transform(centers)
            
        elif self.which == 4:  # Hyperbolic Tangent + Log Transform on R5
            # preprocessing, we can assume that if we already have centroids that were fitted on this normalization, then we don't need to apply the log
            # transformation to R5 on the centroids because the centroids literally already have that fitted on them :)
            try:
                df_copy = data.copy()
                df_copy['R5'] = np.log(df_copy['R5'])
            except:
                raise Exception("Missing R5!")

            data = df_copy.to_numpy()

            self.norm_name = 'r5log_'
            self.s = StandardScaler()
            self.s2 = MinMaxScaler()

            data = self.s.fit_transform(data)
            data = np.tanh(data)
            data = self.s2.fit_transform(data)

            if centers is not None:
                centers = self.s.transform(centers)
                centers = np.tanh(centers)
                centers = self.s2.transform(centers)

        self.data = data

        if centers is not None:
            self.centers = centers
            return self.data, self.centers

        return self.data

    def denorm(self, centers=None, new_data=None):
        if new_data is not None:
            self.data = new_data

        if centers is not None:
            self.centers = centers

        if self.which == 0:    # Min Max Normalization
            self.data = self.s.inverse_transform(self.data)

            if self.centers is not None:
                self.centers = self.s.inverse_transform(self.centers)
        
        elif self.which == 1:  # Robust Scaler Normalization
            self.data = self.s2.inverse_transform(self.data)
            self.data = self.s.inverse_transform(self.data)

            if self.centers is not None:
                self.centers = self.s2.inverse_transform(self.centers)
                self.centers = self.s.inverse_transform(self.centers)

        elif self.which == 2:  # Quantile Transform
            self.data = self.s2.inverse_transform(self.data)
            self.data = self.s.inverse_transform(self.data)

            if self.centers is not None:
                self.centers = self.s2.inverse_transform(self.centers)
                self.centers = self.s.inverse_transform(self.centers)

        elif self.which == 3 or self.which == 4:  # Hyperbolic Tangent
            self.data = self.s2.inverse_transform(self.data)
            self.data = np.arctanh(self.data)
            self.data = self.s.inverse_transform(self.data)

            if self.centers is not None:
                self.centers = self.s2.inverse_transform(self.centers)
                self.centers = np.arctanh(self.centers)
                self.centers = self.s.inverse_transform(self.centers)
        
        if self.centers is not None:
            return self.data, self.centers
        
        return self.data
