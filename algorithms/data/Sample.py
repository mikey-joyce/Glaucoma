# Standard library imports
import itertools
import random
import time

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm

# Local imports
from Retina import RetinaModel


basic_feats = [
    'IOP', 'MAP', 'SBP', 'DBP', 'HR', 'Qmean', 'P1', 'P2', 'P4', 'P5', 'R4', 'R5'
]


class Sample():
    def __init__(self, 
                 n_samples: int = 10, 
                 feats: list = basic_feats, 
                 d_dir: str = '../../data/mssm/',
                 file_path: str = '../../data/mssm/distributed_sampling/example.csv',
                 fn: str = 'sinai.csv'):
        # initialize variables
        self.n_samples = n_samples
        self.features = feats

        self.data_dir = d_dir

        if not file_path.lower().endswith('.csv'):
            raise Exception("File path is not a csv!")
        
        self.file_path = file_path

        data = pd.read_csv(self.data_dir + fn)

        # start the pipeline
        self.data, self.data2augment = self.__preprocess(data)
        self.result = self.__sample()

        if self.result is None:
            raise Exception("Pipline failed!")

    def compute_map(self, sbp, dbp):
        """Compute Mean Arterial Pressure (MAP)."""
        return dbp + (1/3) * (sbp - dbp)

    def __sample(self):
        start = time.time()

        resulting_df = pd.DataFrame(columns=self.features)

        for i, row in tqdm(self.data2augment.iterrows(), total=self.data2augment.shape[0]):
            curr_iop = row['IOP']
            curr_sbp = row['SBP']
            curr_dbp = row['DBP']
            curr_hr = row['HR']

            length = self.n_samples
            iop_range, pressure_range, hr_range = 3, 10, 10
            iop_samples = [float(curr_iop + random.randint(-iop_range, iop_range)) for _ in range(length)]
            sbp_samples = [float(curr_sbp + random.randint(-pressure_range, pressure_range)) for _ in range(length)]
            dbp_samples = [float(curr_dbp + random.randint(-pressure_range, pressure_range)) for _ in range(length)]
            hr_samples = [float(curr_hr + random.randint(-hr_range, hr_range)) for _ in range(length)]

            map_samples = []
            for j in range(len(sbp_samples)):
                map_samples.append(self.compute_map(sbp_samples[j], dbp_samples[j]))

            temp = pd.DataFrame({
                'IOP': iop_samples,
                'MAP': map_samples,
                'SBP': sbp_samples,
                'DBP': dbp_samples,
                'HR': hr_samples
            })

            rm = RetinaModel(
                icp='../../config/Initial_wIOP.csv',
                fn=temp,
                ret_fn=0
            )
            res = rm.pipeline()

            res = res.drop(columns=['R1'])
            row_df = pd.DataFrame([row], columns=self.features)
            temp_df = pd.concat([row_df, res], ignore_index=True)

            resulting_df = pd.concat([resulting_df, temp_df], ignore_index=True)

        end = time.time()
        elapsed = end - start
        print(f"Elapsed time: {elapsed}s")
        
        return resulting_df

    def save(self):
        self.result.to_csv(self.file_path, index=False)

    def __preprocess(self, data):
        train_X = data[self.features]
        train_X = train_X.dropna()

        data = data.drop(index=259)
        train_X = train_X.drop(index=259) # for some reason this patient seems to be bad

        data = data.drop(index=279)
        train_X = train_X.drop(index=279)

        return data, train_X


if __name__ == '__main__':
    ex = Sample(n_samples=1)
    ex.save()

