# Helper functions for the IGPS dataset
import pandas as pd

def get_12Dfeatures(data):
    '''try:
        features = [
            'IOP', 'MAP', 'BPSYS', 'BPDYS', 'Qmean', 'P1mean', 'P2mean', 'P4mean', 'P5mean', 'R4mean', 'R5amean',
            'R5bmean'
        ]
        new = data[features].copy()
        new = new.rename(columns={'BPSYS': 'SBP', 'BPDYS': 'DBP'})
    except:
        try:
            features = [
                'Patient', 'IOP', 'MAP', 'SBP', 'DBP', 'HR', 'Q', 'P1', 'P2', 'P4', 'P5', 'R4', 'R5'
            ]
            new = data[features].copy()
            new = new.rename(columns={'Q': 'Qmean', 'P1': 'P1mean', 'P2': 'P2mean',
                                      'P4': 'P4mean', 'P5': 'P5mean', 'R4amean': 'R4mean'})
        except:
            features = [
                'IOP', 'MAP', 'SBP', 'DBP', 'Q', 'P1', 'P2', 'P4', 'P5', 'R4', 'R5amean',
                'R5bmean'
            ]
            new = data[features].copy()
            new = new.rename(columns={'Q': 'Qmean', 'P1': 'P1mean', 'P2': 'P2mean',
                                      'P4': 'P4mean', 'P5': 'P5mean', 'R4': 'R4mean'})'''

    features = [
        'Patient', 'IOP', 'MAP', 'SBP', 'DBP', 'HR', 'Q', 'P1', 'P2', 'P4', 'P5', 'R4', 'R5'
    ]

    new = data[features].copy()

    return new


def import_dataset(file_name, flag=False):
    # THIS IS A VERY SPECIFIC FUNCTION TO THE FILES DAPHNE SENT ME
    # Enhanced_IGPS_First_Visits_HR=60_Labels.xlsx
    # Enhanced_IGPS_First_Visits_realHR_Labels.xlsx
    if flag is True:
        data = pd.read_csv(file_name)
    else:
        data = pd.read_excel(file_name)
    data, labels = data.iloc[:, :-1], data.iloc[:, -1]

    later = data.copy()
    table = data[data.columns.intersection(['IOP', 'MAP'])]
    labels = labels.to_numpy()

    return data, labels, table, later