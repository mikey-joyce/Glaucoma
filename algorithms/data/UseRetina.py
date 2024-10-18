import pandas as pd

from Sample import Sample
from Retina import RetinaModel

data_to_enhance = '../../data/seed/SEED_progression.csv'
data_to_enhance = pd.read_csv(data_to_enhance)

sbp_samples = data_to_enhance['SBP']
dbp_samples = data_to_enhance['DBP']

def compute_map(sbp, dbp):
    """Compute Mean Arterial Pressure (MAP)."""
    return dbp + (1/3) * (sbp - dbp)

map_samples = []
for j in range(len(sbp_samples)):
    map_samples.append(compute_map(sbp_samples[j], dbp_samples[j]))

temp = pd.DataFrame({
    'Patient': data_to_enhance['sno'],
    'IOP': data_to_enhance['IOP'],
    'MAP': map_samples,
    'SBP': sbp_samples,
    'DBP': dbp_samples,
    'HR': data_to_enhance['HR']
})

rm = RetinaModel(
    icp='../../config/Initial_wIOP.csv',
    fn=temp,
    ret_fn=0
)
res = rm.pipeline()

res.to_csv('../../data/seed/SEED_progression_enhanced.csv', index=False)