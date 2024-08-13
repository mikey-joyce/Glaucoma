import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
import warnings
warnings.filterwarnings('ignore')

import sys
import numpy
import numpy as np
import math
import scipy
from scipy.optimize import fsolve
from scipy.optimize import root
import pandas as pd
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from pathlib import Path

# for mac
# import matplotlib
# matplotlib.use('PS')

from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
import pickle
import statistics


class RetinaModel:
    def __init__(self, icp, fn, ret_fn):
        self.init_cond = pd.read_csv(icp)
        self.filename = fn
        self.return_file = ret_fn

        self.Pin = [];
        self.T = [];
        self.Q1d = [];
        self.Q4a = [];
        self.Q4b = [];
        self.Q5a = [];
        self.Q5b = [];
        self.R1c = [];
        self.R1d = [];
        self.R4a = [];
        self.R4b = [];
        self.R5a = [];
        self.R5b = [];

        # Idea behind what you are doing:
        # 1) take the dataset and filter out just the header in which you are
        #    interested to create the new dataset
        # 2) perform an apply and launch for each row the function
        # 3) relax and let the code to the rest

        # ====================
        # Load Model Parameter
        # ====================

        # Output pressure of the lumped modelp
        self.P_out = 14.00

        # Fixed Resistances [mm Hg s/mL] - Table 2
        self.R_in = 2.25e4
        self.R_1a = 4.30e3
        self.R_1b = 4.30e3
        self.R_2a = 6.00e3
        self.R_2b = 6.00e3
        self.R_3a = 5.68e3
        self.R_3b = 5.68e3
        self.R_5c = 1.35e3
        self.R_5d = 1.35e3
        self.R_out = 5.74e3
        self.R_constT = self.R_1a + self.R_1b + self.R_2a + self.R_2b + self.R_3a + self.R_3b + self.R_5c + self.R_5d

        # Parameters Values for variable resistances

        self.v = 0.49  # Wall Poission's ratio (Table 1/Table 9)

        # CRA parameter values
        self.mu_CRA_cP = 3.0  # fluid viscosity [cP] - Table 1
        self.mu_CRA = self.mu_CRA_cP * 1e-3 * 0.00750062  # fluid viscosity: conversion from [cP] to [mmHg s]
        self.E_CRA_MPa = 0.3  # Young Modulus [MPa] - Table 1
        self.E_CRA = self.E_CRA_MPa * 1e6 * 0.00750062  # Young Modulus: conversion from [MPa] to [mmHg]
        self.d_CRA_ref = 175e-4  # Refernce radius [cm] - Table 1
        self.r_CRA_ref = self.d_CRA_ref / 2
        self.A_ref_CRA = math.pi * self.r_CRA_ref ** 2  # Reference cross sectional area [cm^2]
        self.h_CRA = 39.7239e-4  # Thickness [cm] of vessel wall - Table 1
        self.L_1c = 0.2e-1  # Lengths [cm] of segment c - Table 1
        self.L_1d = 1e-1  # Lengths [cm] of segment d - Table 1
        self.K_l_CRA = 12 * self.A_ref_CRA / (math.pi * self.h_CRA ** 2)  # [-]
        self.K_p_CRA = (1 / 12) * (self.E_CRA * self.h_CRA ** 3 / (1 - self.v ** 2)) * (math.pi / self.A_ref_CRA) ** (3 / 2)  # [mmHg ]
        self.k0_CRA_1c = (self.A_ref_CRA ** 2) / (8 * math.pi * self.mu_CRA * self.L_1c)
        self.k0_CRA_1d = self.A_ref_CRA ** 2 / (8 * math.pi * self.mu_CRA * self.L_1d)

        # CRV parameter values
        self.mu_CRV_cP = 3.24  # fluid viscosity [cP] - Table 1
        self.mu_CRV = self.mu_CRV_cP * 1e-3 * 0.00750062  # fluid viscosity: conversion from [cP] to [mmHg s]
        self.E_CRV_MPa = 0.6  # Young Modulus [MPa] - Table 1
        self.E_CRV = self.E_CRV_MPa * 1e6 * 0.00750062  # Young Modulus: conversion from [MPa] to [mmHg] Table 1
        self.d_CRV_ref = 238e-4  # [cm]
        self.r_CRV_ref = self.d_CRV_ref / 2  # [cm]
        self.A_ref_CRV = math.pi * self.r_CRV_ref ** 2  # Reference cross sectional area [cm^2]
        self.h_CRV = 10.7e-4  # wall thickness [cm] - Table 1
        self.L_5a = 1e-1  # Length of segment a [cm] - Table 1
        self.L_5b = 0.2e-1  # Length of segment b [cm] - Table 1
        self.K_l_CRV = 12 * self.A_ref_CRV / (math.pi * self.h_CRV ** 2)
        self.K_p_CRV = (1 / 12) * (self.E_CRV * self.h_CRV ** 3 / (1 - self.v ** 2)) * (math.pi / self.A_ref_CRV) ** (3 / 2)
        self.k0_CRV_5a = self.A_ref_CRV ** 2 / (8 * math.pi * self.mu_CRV * self.L_5a)
        self.k0_CRV_5b = self.A_ref_CRV ** 2 / (8 * math.pi * self.mu_CRV * self.L_5b)

        # venules parameter values
        self.mu_VEN_cP = 3.24  # fluid viscosity [cP] - Matlab File
        self.mu_VEN = self.mu_VEN_cP * 1e-3 * 0.00750062  # fluid viscosity: conversion from [cP] to [mmHg s]
        self.E_VEN_MPa = 0.066  # Young Modulus [Mpa] -  Table 9
        self.E_VEN = self.E_VEN_MPa * 1e6 * 0.00750062  # Young Modulus: conversion from [Pa] to [mmHg]
        self.d_VEN_ref = 0.015476685724901  # equivalent diameter [cm] - Matlab File
        self.r_VEN_ref = self.d_VEN_ref / 2
        self.A_ref_VEN = math.pi * self.r_VEN_ref ** 2  # Reference cross sectional area [cm^2]
        self.h_VEN = self.d_VEN_ref / 20;  # wall thickness [cm] -  Table 9
        self.L_VEN = 0.413520;  # equivalent length [cm] - Matlab File
        self.L_4a = self.L_VEN / 2;  # [cm]
        self.L_4b = self.L_VEN / 2;  # [cm]
        self.K_l_VEN = 12 * self.A_ref_VEN / (math.pi * self.h_VEN ** 2)
        self.K_p_VEN = (1 / 12) * (self.E_VEN * self.h_VEN ** 3 / (1 - self.v ** 2)) * (math.pi / self.A_ref_VEN) ** (3 / 2)
        self.k0_VEN_4a = self.A_ref_VEN ** 2 / (8 * math.pi * self.mu_VEN * self.L_4a)
        self.k0_VEN_4b = self.A_ref_VEN ** 2 / (8 * math.pi * self.mu_VEN * self.L_4b)

        # definition of capacitance

        self.C1 = 7.22 * 10 ** (-7)  # mL/mmHg
        self.C2 = 7.53 * 10 ** (-7)  # mL/mmHg
        self.C4 = 1.67 * 10 ** (-5)  # mL/mmHg
        self.C5 = 1.07 * 10 ** (-5)  # mL/mmHg

    def Pin_Waveform(self, time, c):
        global Thr
        # Definition Variable
        SP = c[0]
        DP = c[1]
        HR = c[3]
        Thr = 60 / HR;
        Tm = time % Thr;

        # Computation Pin

        if Tm <= 0.082 * Thr:
            v = 0.65 * SP - 0.475 * DP * math.sin((2 * math.pi * Tm / (4 * 0.082 * Thr))
                                                  + (2 * math.pi * 0.082 * Thr / (0.328 * Thr)))

        elif ((Tm > 0.082 * Thr) and (Tm <= 0.112 * Thr)):
            v = 0.65 * SP + 0.9 * math.sin((2 * math.pi * Tm / (0.03 * Thr))
                                           - (2 * math.pi * 0.082 * Thr / (0.03 * Thr)))

        elif ((Tm > 0.112 * Thr) and (Tm <= 0.398 * Thr)):
            v = 0.65 * SP + 0.118 * SP * math.sin((2 * math.pi * Tm / (0.572 * Thr))
                                                  - (2 * math.pi * 0.112 * Thr / (0.572 * Thr)))

        elif ((Tm > 0.398 * Thr) and (Tm <= 0.432 * Thr)):
            v = -(0.13 * SP * Tm / (0.034 * Thr)) + 0.65 * SP + (0.13 * SP * 0.398 * Thr / (0.034 * Thr))

        elif ((Tm > 0.432 * 60 / HR) and (Tm <= 0.482 * 60 / HR)):
            v = (0.52 * SP - 0.8 * math.sin(
                (2 * math.pi * Tm / (0.05 * Thr)) + (2 * math.pi * 0.332 * Thr / (0.05 * Thr))))

        else:
            v = (0.52 * SP + (0.52 * SP - 0.5 * DP) * math.sin(
                (2 * math.pi * Tm / (2.072 * Thr)) + (2 * math.pi * 0.554 * Thr / (2.072 * Thr))))

        return v

    def f(self, time, P, c):
        global R_4a, R_4b, R_5a, R_5b, QR
        global Pin, Q1d, Q4a, Q4b, Q5a, Q5b

        # Define initial condition
        P1 = P[0]
        P2 = P[1]
        P4 = P[2]
        P5 = P[3]

        # Save Global variable of interest

        self.T.append(time)
        self.Pin.append(self.Pin_Waveform(time, c))

        # definition of capacitance

        C1 = 7.22 * 10 ** (-7)  # mL/mmHg
        C2 = 7.53 * 10 ** (-7)  # mL/mmHg
        C4 = 1.67 * 10 ** (-5)  # mL/mmHg
        C5 = 1.07 * 10 ** (-5)  # mL/mmHg

        # definition of transmural pressures   (c[2] is IOP)

        deltaP1c = P1 - c[2]
        deltaP1d = P1 - c[2]
        deltaP4a = P4 - c[2]
        deltaP4b = P4 - c[2]
        deltaP5a = P5 - c[2]
        deltaP5b = P5 - c[2]

        # definition of variable resistances - CRA

        R_1c = (1 / self.k0_CRA_1c) * (1 + deltaP1c / (self.K_p_CRA * self.K_l_CRA)) ** (-4)
        R_1d = (1 / self.k0_CRA_1d) * (1 + deltaP1d / (self.K_p_CRA * self.K_l_CRA)) ** (-4)

        # definition of variable resistances - venules and CRV

        if deltaP4a >= 0:
            R_4a = (1 / self.k0_VEN_4a) * (1 + deltaP4a / (self.K_p_VEN * self.K_l_VEN)) ** (-4)
        else:
            R_4a = (1 / self.k0_VEN_4a) * (1 - deltaP4a / (self.K_p_VEN)) ** (4 / 3)

        if deltaP4b >= 0:
            R_4b = (1 / self.k0_VEN_4b) * (1 + deltaP4b / (self.K_p_VEN * self.K_l_VEN)) ** (-4)
        else:
            R_4b = (1 / self.k0_VEN_4b) * (1 - deltaP4b / (self.K_p_VEN)) ** (4 / 3)

        if deltaP5a >= 0:
            R_5a = (1 / self.k0_CRV_5a) * (1 + deltaP5a / (self.K_p_CRV * self.K_l_CRV)) ** (-4)
        else:
            R_5a = (1 / self.k0_CRV_5a) * (1 - deltaP5a / (self.K_p_CRV)) ** (4 / 3)

        if deltaP5b >= 0:
            R_5b = (1 / self.k0_CRV_5b) * (1 + deltaP5b / (self.K_p_CRV * self.K_l_CRV)) ** (-4)
        else:
            R_5b = (1 / self.k0_CRV_5b) * (1 - deltaP5b / (self.K_p_CRV)) ** (4 / 3)

        # definition of nonlinear system

        F1 = ((self.Pin_Waveform(time, c) - P1) / (self.R_in + self.R_1a) - (P1 - P2) / (self.R_1b + R_1c + R_1d + self.R_2a)) / C1
        F2 = ((P1 - P2) / (self.R_1b + R_1c + R_1d + self.R_2a) - (P2 - P4) / (self.R_2b + self.R_3a + self.R_3b + R_4a)) / C2
        F3 = ((P2 - P4) / (self.R_2b + self.R_3a + self.R_3b + R_4a) - (P4 - P5) / (R_4b + R_5a + R_5b + self.R_5c)) / C4
        F4 = ((P4 - P5) / (R_4b + R_5a + R_5b + self.R_5c) - (P5 - self.P_out) / (self.R_5d + self.R_out)) / C5

        # ==================================================
        # Saving Part for resistances and  fluxes
        # ==================================================

        # =====================

        # Resistances

        self.R1c.append(R_1c);
        self.R1d.append(R_1d);
        self.R4a.append(R_4a);
        self.R4b.append(R_4b);
        self.R5a.append(R_5a);
        self.R5b.append(R_5b);

        # =====================
        # Fluxes

        self.Q1d.append(deltaP1d / R_1d);
        self.Q4a.append(deltaP4a / R_4a);
        self.Q4b.append(deltaP4b / R_4b);
        self.Q5a.append(deltaP5a / R_5a);
        self.Q5b.append(deltaP5b / R_5b);

        # final definition of function

        F = [F1, F2, F3, F4]

        # returns the calculated derivatives of the state variables F1-F4. i.e. the rate of change of these variables.
        return F  # F are the differentiation

    def Ind_Input(self):
        # Load the file
        #current_path = Path.cwd()
        #database_folder = current_path
        #file = database_folder / filename
        data = pd.read_csv(self.filename)
        return data

    def sel_InCond(self, IOP):

        # Load the file containing the initial condition
        #current_path = Path.cwd()
        dataIOP = self.init_cond

        FilterIOP = dataIOP[dataIOP['IOP'] == np.rint(IOP)]

        P1 = float(FilterIOP['P1'])
        P2 = float(FilterIOP['P2'])
        P4 = float(FilterIOP['P4'])
        P5 = float(FilterIOP['P5'])
        P0 = [P1, P2, P4, P5]
        return P0

    def PostProcessing(self, t, P, c):
        # Computation of the output utilizing vectorized function to speed up the code.
        # Computation of the input pressure
        Pin = [self.Pin_Waveform(i, c) for i in t]

        # Delta pressures

        deltaP1c = P[0] - c[2]
        deltaP1d = P[0] - c[2]
        deltaP4a = P[2] - c[2]
        deltaP4b = P[2] - c[2]
        deltaP5a = P[3] - c[2]
        deltaP5b = P[3] - c[2]

        # Computation of the resistances

        R_1c = [(1 / self.k0_CRA_1c) * (1 + dp1c / (self.K_p_CRA * self.K_l_CRA)) ** (-4) for dp1c in deltaP1c]
        R_1d = [(1 / self.k0_CRA_1d) * (1 + dp1d / (self.K_p_CRA * self.K_l_CRA)) ** (-4) for dp1d in deltaP1d]

        R_4a = [(1 / self.k0_VEN_4a) * (1 + dp4a / (self.K_p_VEN * self.K_l_VEN)) ** (-4) if dp4a >= 0 else
                (1 / self.k0_VEN_4a) * (1 - dp4a / (self.K_p_VEN)) ** (4 / 3) for dp4a in deltaP4a]
        R_4b = [(1 / self.k0_VEN_4a) * (1 + dp4b / (self.K_p_VEN * self.K_l_VEN)) ** (-4) if dp4b >= 0 else
                (1 / self.k0_VEN_4a) * (1 - dp4b / (self.K_p_VEN)) ** (4 / 3) for dp4b in deltaP4b]
        R_5a = [(1 / self.k0_CRV_5a) * (1 + dp5a / (self.K_p_CRV * self.K_l_CRV)) ** (-4) if dp5a >= 0 else
                (1 / self.k0_CRV_5a) * (1 - dp5a / (self.K_p_CRV)) ** (4 / 3) for dp5a in deltaP5a]
        R_5b = [(1 / self.k0_CRV_5b) * (1 + dp5b / (self.K_p_CRV * self.K_l_CRV)) ** (-4) if dp5b >= 0 else
                (1 / self.k0_CRV_5b) * (1 - dp5b / (self.K_p_CRV)) ** (4 / 3) for dp5b in deltaP5b]

        # Computation of the volume flow rates

        Qin = [(self.Pin_Waveform(t, c) - p1) / (self.R_in + self.R_1a) for t, p1 in zip(t, P[0])]
        Q12 = [(p1 - p2) / (self.R_1b + r1c + r1d + self.R_2a) for (p1, p2, r1c, r1d) in zip(P[0], P[1], R_1c, R_1d)]
        Q24 = [(p2 - p4) / (self.R_2b + self.R_3a + self.R_3b + r4a) for (p2, p4, r4a) in zip(P[1], P[2], R_4a)]
        Q45 = [(p4 - p5) / (r4b + r5a + r5b + self.R_5c) for (p4, p5, r4b, r5a, r5b) in zip(P[2], P[3], R_4b, R_5a, R_5b)]
        Qout = [(p5 - self.P_out) / (self.R_5d + self.R_out) for p5 in P[3]]

        #     print(len(Qin))
        # Compute of the variables needed for RNN analysis

        F1 = [(qin - q12) / self.C1 for (qin, q12) in zip(Qin, Q12)]
        F2 = [(q12 - q24) / self.C2 for (q12, q24) in zip(Q12, Q24)]
        F3 = [(q24 - q45) / self.C4 for (q24, q45) in zip(Q24, Q45)]
        F4 = [(q45 - qout) / self.C5 for (q45, qout) in zip(Q45, Qout)]

        # Organize the variable into list
        F = {'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4};
        Q = {'Qin': np.asarray(Qin), 'Q12': np.asarray(Q12), 'Q24': np.asarray(Q24), 'Q45': np.asarray(Q45),
             'Qout': (Qout)}
        R = {'R_1c': np.asarray(R_1c), 'R_1d': np.asarray(R_1d), 'R_4a': np.asarray(R_4a), 'R_4b': np.asarray(R_4b),
             'R_5a': np.asarray(R_5a), 'R_5b': np.asarray(R_5b)}

        return Pin, F, Q, R

    def Shimpatica_Func(self, row):
        P0 = self.sel_InCond(row['IOP'])
        Pinput = [row['SBP'],
                  row['DBP'],
                  row['IOP'],
                  row['HR']]

        Tc = 60 / Pinput[-1];
        Tfin = Tc * 10

        # f(t, y, Pinput) is the lambda function that represents the ODEs required to be solved where
        # t = current time, y = state variable, Pinput = input parameter

        fR = solve_ivp(lambda t, y: self.f(t, y, Pinput), (0, Tfin), P0, method='BDF', atol=1e-8, rtol=1e-8)
        if fR.success == True:
            # print('\n###########################')
            # print('The simulation was successfull:', fR.success)
            # print('###########################')
            pass
        else:
            print('\n###########################')
            print('The simulation was not successful:', fR.success)
            print('###########################')

        Pin, F, Q, R = self.PostProcessing(fR.t, fR.y, Pinput)

        # Now complete the quantity for both
        # the stationary and non stationary

        indexLastCycle = np.where(fR.t > Tc * 9)

        # Select the variable you want to save

        HeadP = ['P1', 'P2', 'P4', 'P5'];
        HeadR = ['R_1c', 'R_1d', 'R_4a', 'R_4b', 'R_5a', 'R_5b'];
        HeadQ = ['Q12', 'Q24', 'Q45']  # Header of interest

        timeLastCycle = fR.t[indexLastCycle]

        valP = [[min(item[indexLastCycle]), max(item[indexLastCycle]),
                 np.trapz(item[indexLastCycle], x=timeLastCycle) / (timeLastCycle[-1] - timeLastCycle[0])] for item in
                fR.y]
        timeP = [[np.argmin(item[indexLastCycle]), np.argmax(item[indexLastCycle])] for item in fR.y]

        # Compute resistances and Q quantities with list comprehension
        # to speed up process and make it more pythonic

        valQ = [[min(Q[item][indexLastCycle]),
                 max(Q[item][indexLastCycle]),
                 np.trapz(Q[item][indexLastCycle], x=timeLastCycle) / (timeLastCycle[-1] - timeLastCycle[0])]
                for item in HeadQ]

        timeQ = [[np.argmin(Q[item][indexLastCycle]),
                  np.argmax(Q[item][indexLastCycle])]
                 for item in HeadQ]

        # Compute resistances and Q quantities with list comprehension
        # to speed up process and make it more pythonic

        valR = [[min(R[item][indexLastCycle]),
                 max(R[item][indexLastCycle]),
                 np.trapz(R[item][indexLastCycle], x=timeLastCycle) / (timeLastCycle[-1] - timeLastCycle[0])]
                for item in HeadR]

        timeR = [[np.argmin(R[item][indexLastCycle]),
                  np.argmax(R[item][indexLastCycle])]
                 for item in HeadR]

        # Extract the variable of interest from the list
        P1sys, P2sys, P4sys, P5sys = [item[1] for item in valP]
        P1dis, P2dis, P4dis, P5dis = [item[0] for item in valP]
        P1mean, P2mean, P4mean, P5mean = [item[2] for item in valP]
        Q12sys, Q24sys, Q45sys = [item[1] for item in valQ]
        Q12dis, Q24dis, Q45dis = [item[0] for item in valQ]
        Q12mean, Q24mean, Q45mean = [item[2] for item in valQ]

        Qmean = (Q12mean + Q24mean + Q45mean) / 3.0

        #     Qmean_total = Q12mean + Q24mean + Q45mean

        R1csys, R1dsys, R4asys, R4bsys, R5asys, R5bsys = [item[0] for item in valR]
        R1cdis, R1ddis, R4adis, R4bdis, R5adis, R5bdis = [item[1] for item in valR]
        R1cmean, R1dmean, R4amean, R4bmean, R5amean, R5bmean = [item[2] for item in valR]

        R5 = np.trapz([R5amean, R5bmean, self.R_5c, self.R_5d])
        R1 = np.trapz([R1cmean, R1dmean, self.R_1a, self.R_1b])

        R4 = R4amean + R4bmean

        #     R4sys = R4asys + R4bsys
        #     R4dis = R4adis + R4bdis
        #     R4mean = R4amean + R4bmean

        # Plot the variable of interest to verify that you are doing the right thing
        # for idx,i in enumerate(HeadRQ):
        # plt.plot(res['TimeRQ'][indexQR[0]:indexQR[-1]],res[i][indexQR[0]:indexQR[-1]])
        # plt.plot(res['TimeRQ'][indexQR[0] + timeQR[idx][0]], valQR[idx][0], 'o')
        # plt.plot(res['TimeRQ'][indexQR[0] + timeQR[idx][1]], valQR[idx][1],'o')
        # # if ind == 0:
        # plt.show()

        return P1mean, P2mean, P4mean, P5mean, Qmean, R1, R4, R5

    def pipeline(self):
        if type(self.filename) is str:
            data = self.Ind_Input()
        else:
            data = self.filename
        datamissingRows = []

        orig_len = len(data)
        # =============================
        # Identify if there are any Nan
        # =============================

        for row in data.itertuples():
            if (pd.isna(row.SBP) == True or \
                    pd.isna(row.DBP) == True or
                    pd.isna(row.IOP) == True or
                    pd.isna(row.HR) == True):
                print('subject %d has a missing value' % (row.Index))
                datamissingRows.append(row.Patient);
                data.drop(row.Index, inplace=True)

        # Verifiy that everything has been done correctly

        # if len(datamissingRows) + len(data) == orig_len:
        #    print('\n ## It seems good ##\n')

        # tqdm.pandas(desc=None)
        # res = data.progress_apply(self.Shimpatica_Func, axis=1, result_type='expand')

        # above comments replaced with this snippet
        f_P1mean, f_P2mean, f_P4mean, f_P5mean, f_Qmean, f_R1, f_R4, f_R5 = [], [], [], [], [], [], [], []
        for _, row in data.iterrows():
            temp_res = self.Shimpatica_Func(row)
            
            f_P1mean.append(temp_res[0])
            f_P2mean.append(temp_res[1])
            f_P4mean.append(temp_res[2])
            f_P5mean.append(temp_res[3])
            f_Qmean.append(temp_res[4])
            f_R1.append(temp_res[5])
            f_R4.append(temp_res[6])
            f_R5.append(temp_res[7])

        res = pd.DataFrame({
            'P1mean': f_P1mean, 
            'P2mean': f_P2mean, 
            'P4mean': f_P4mean, 
            'P5mean': f_P5mean, 
            'Qmean': f_Qmean, 
            'R1': f_R1, 
            'R4': f_R4, 
            'R5': f_R5
        })

        db_enhanced = pd.concat([data, res], axis=1)

        if type(self.return_file) is str:
            db_enhanced.to_csv(self.return_file, index=False)
            print('Done!')
        else:
            print('Done!')
            return db_enhanced
