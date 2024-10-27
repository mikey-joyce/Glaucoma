'''
This file is meant for computations that I need that can be redundant to keep copy and pasting into notebooks
'''

def mean_arterial_pressure(systolic_bp, diastolic_bp):
    map_value = (1 / 3) * (systolic_bp - diastolic_bp) + diastolic_bp
    return map_value