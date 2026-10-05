# -*- coding: utf-8 -*-
"""DAC-to-SNG process flowsheet (thesis Section 8.2) as a Pyomo grey-box model.

The dual-function-material (DFM) capture and methanation unit is the truth model,
`TRF_blackbox`, embedded through Pyomo ExternalFunctions with finite-difference
sensitivities (`grad_TRF_blackbox`); the rest of the flowsheet is glass-box.

Importing this module builds the model `m`. The trust-region degrees of freedom are
`DEGREES_OF_FREEDOM(m)` and the surrogate basis is `basis_rule` (zero basis, so the
surrogate is the first-order corrected Taylor model of thesis (6.1)).

The model equations are those of the script that produced the thesis results; only the
imports, the location of the data file and the solve section differ.
"""
import math
import os
import sys

import numpy as np
from pyomo.environ import *

_HERE = os.path.dirname(os.path.abspath(__file__))

class Tee:
    """Custom class to write to both console and file simultaneously."""
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout  # Save original stdout

    def write(self, message):
        self.stdout.write(message)  # Print to console
        self.file.write(message)    # Write to file

    def flush(self):
        self.stdout.flush()
        self.file.flush()

Streams = {1,11,12,13,14,15,16,2,21,22,23,24,25,26,27,28,29,3,4,5,51,52,6,61,62,63,64,65,66,635,636,665,666,7,8,9,91,92,93,94,95,96,10,20}

Initial_Streams = {1: {'CO2':26, 'H2':0.0, 'CH4':0.0, 'H2O':0.0, 'N2':51226, 'O2':13747}}


# MW unit is kg/kmole, ro = kg/m3, cp = kJ/kgmole-C @350 C
Components = {
    'CO2': {'MW':44.01, 'cost':0.0, 'price':0.0, 'ro':1.84, 'cp':47.25},
    'H2':  {'MW':2.016, 'cost':0.0, 'price':-200, 'ro':0.0837, 'cp':29.29}, # Hydrogen density @350
    'CH4': {'MW':16.04, 'cost':0.0, 'price':9.0, 'ro':0.657, 'cp':53.78},
    'H2O': {'MW':18.01528, 'cost':0.0, 'price':0.0, 'ro':0.88, 'cp':37.21},  # Water vapor density @ 350
    'N2':  {'MW':28.0134, 'cost':0.0, 'price':0.0, 'ro':1.165, 'cp':0},
    'O2':  {'MW':31.999, 'cost':0.0, 'price':0.0, 'ro':1.331, 'cp':0},
}

# H2O --> H2 + 1/2 O2
Stoich_ratio1 = {'CO2':0, 'H2':1, 'CH4':0, 'H2O':-1, 'N2':0, 'O2':0.5}

# CO2 + 4H2 --> CH4 + 2H2O
Stoich_ratio2 = {'CO2':-1,'H2':-4,'CH4':1,'H2O':2,'N2':0,'O2':0}

# Capacity (b) in m3/s Fan maximum head in pa (Seider et al., 2017) # Vane-axial_fan
Air_intake_units_Q_min = 0.4722
Air_intake_units_Q_max = 377.7778
Air_intake_units_Fan_maxH = 4000

# Costing ref.: Towler & Sinnott, 2021 , Ref. for Air-Cooler: Seider et al., 2017
# lower limits to be corrected with modifications in equations to enforce the selection of S_lower as min.
equipment_size_ref = {
    'Compressor_Centrifugal'  : {'S_lower' : 1 ,'S_upper' : 30000 , 'a' : 490000 , 'b' : 16800 , 'n' : 0.6},    # kW
    'Pump'                  : {'S_lower' : 1 ,'S_upper' : 2500 , 'a' : 950 , 'b' : 1770 , 'n' : 0.6},           # kW
    'Axial_Fan'             : {'S_lower' : 100 , 'S_upper' : 170000 , 'a' : 4200 , 'b' : 27 , 'n' : 0.8},       # m3/h
    'Centrifugal_Fan'       : {'S_lower' : 100 , 'S_upper' : 170000 , 'a' : 53000 , 'b' : 28000 , 'n' : 0.8},   # m3/h
    'HEX_Shell&Tube'        : {'S_lower' : 10 , 'S_upper' : 1000 , 'a' : 24000 , 'b' : 46 , 'n' : 1.2},         # m2
    'Furnace_Cylindrical'   : {'S_lower' : 0.2 , 'S_upper' : 60 , 'a' : 68500 , 'b' : 93000 , 'n' : 0.8},       # MW
    'Reactor'               : {'S_lower' : 0.5 , 'S_upper' : 100 , 'a' : 53000 , 'b' : 28000 , 'n' : 0.8},      # m3
    'Air-Cooler'            : {'S_lower' : 40 , 'S_upper' : 150 , 'a' : 0 , 'b' : 2835 , 'n' : 0.45},           # ft2
    'Vertical_Vessel'       : {'S_lower' : 160 , 'S_upper' : 250000 , 'a' : 10000 , 'b' : 29 , 'n' : 0.85}      # kg
    }

M = 1e10                    #BigM
pi = 22/7

T_amb = 293                 # ambient temperature K
T_air_In = 25               # ambient temperature °C
T_product = 40              # Final product temperature

air_Mu = 1.85e-05 #average dynamic viscosity Pa.s
air_ro = 1.185 #average air density kg/m3
air_MW = 29 #average air molecular weight kg/kmol

p_fuel = 5.6869e-06         # $/kJ -- Towler et. al
n_boiler = 0.8              # Boiler efficiency
n_turb = 0.85               # Steam turbine efficiency, for steam pricing calculations
p_bfw = 0.0                 # $/kmol of boiler feed water
kWh_cost = 0.05             # [$/kWh] Electricity cost (Sendi (2022))
W_cost = kWh_cost/3600          # [$/kJ] Electricity cost
Water_cost = 1.1*1.72*1e-3*18.02e-3*1000 # [$/€][€/kg][kg/mol] = [$/kmol] electrolyzer statistisches Bundesamt(link)


Conv_fac1 = 1.0             # Electrolyzers' conversion
SOEL_unit_cost = 269900     # kJ/kmole H2
SOEL_CAPEX = 4200           # $/kW
SOEL_T = 700                # SOEL operating Temperature

Q_Furnace = 275             # kWh heat duty based on outlet H2 @ 350C

# Plant Steam levels C
T_LP = 150
T_MP = 200
T_HP = 250
#U_Boiler = 88.57 #kJ/h-m2-C for Steam-Flue gas system, based on HYSYS calculations ** SHOULD be updated
#U_Boiler = 118 #kJ/h-m2-C for LP Steam-Flue gas system, based on HYSYS calculations
#U_Boiler = 105.7 #kJ/h-m2-C for MP Steam-Flue gas system, based on HYSYS calculations
U_Boiler = 118 #kJ/h-m2-C update Feb. 2024

DT_min = 20   # min, Temperature difference between Hot & Cold streams in HEX

nf_fan = 0.6    # nominal fan efficiency
nm_fan = 0.9    # electrical motor efficiency
FM = 2.5        # Fan material factor (equal to Stainless steel)

# DFM reactor
VR = 7.95e07 #viscous resistance (1/m2)
IR = 7.41  #inertial resistance 1/m
Ri = 0.00105

Conv_fac2 = 1.0             # Methanation reactors' conversion

#DFM_unit_cost = 272 #$/kg DFM 15% Ni,1% Ru,10% K2O,74% CeO2-Al2O3 (20%-80%)
DFM_unit_cost = 156.2 #$/kg DFM 15% Ni,1% Ru,10% K2O,74% CeO2-Al2O3 (20%-80%)
#DFM_unit_cost = 15 # $/kg (based on Sinha et al., 2017 for MIL-101 price          > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
#DFM_SPWeight = 18.511 #kg of DFM on m3 of DFM coat
#DFM_SPWeight = 57.362 #kg of washcoat on m3 of monolith
DFM_SPWeight = 85.43 #kg of washcoat per m3 of monolith (Abdallah et al.,(2023))
#DFM_SPWeight = 210 #kg of washcoat per m3 of monolith (Duyar et al., 2014)
#q_dfm_eq = 0.0014 #Simillar to MIL-101 > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
#q_dfm_eq = 1.14934e-3 #100% of DFM amb. ads. capacity acc. to to Abdallah et al.,(2023) 1%Ru, 10%Na2O/γ-Al2O3 DFM > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
q_dfm_eq = 1.152e-3 #100% of DFM amb. ads. capacity acc. to to Abdallah et al.,(2023) 1%Ru, 10%Na2O/γ-Al2O3 DFM > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
#q_dfm_eq = 6.21e-4 #50% of DFM amb. ads. capacity acc. to to Abdallah et al.,(2023) 1%Ru, 10%Na2O/γ-Al2O3 DFM > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
#q_dfm_eq = 4.9e-4 #kmol CO2 adsorbed per kg of DFM
#q_dfm_eq = 2.05e-3 #kmol CO2 adsorbed per kg of DFM Goldman et al., 2023       > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
#q_dfm_eq = 2.71e-3 #kmol CO2 adsorbed per kg of DFM - Hypothetical value based on MIL-101 results
#k_ads = 0.00023654 # based on curve fitting data on 1%Ru, 10%Na2O/γ-Al2O3 DFM acc. to Abdallah et al.,(2023) > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
k_ads = 0.00025 # based on curve fitting data on 1%Ru, 10%Na2O/γ-Al2O3 DFM acc. to Abdallah et al.,(2023) > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > > >
#k_ads = 0.01 # adsorption rate constant  0.00023654
cp_DFM = 0.718 #kJ/kg.K
T_reaction = 623 #Deg C
DH_ads = 151.79 #NiRuNa/CeAl : -151.79 kJ/mol Enthalpy of adsorption
#DH_ads = 121.29 #NiRuK/CeAl: -121.29 kJ/mol Enthalpy of adsorption
#DH_ads = 31.17 #NiRuCa/CeAl: -31.17 kJ/mol Enthalpy of adsorption
H_methanation = 164 #kJ/mol

#Air-Cooler
#U_Aircooler = 130  #Btu/hr-ft2-F
U_Aircooler = 770   # W/m2-C
L_AC_tube = 40      # Air-Cooler tube length (ft)
AC_tube_gap = 2.5   # Air-Cooler tube spacing (inch)
AC_tube_OD = 1      # OD of the tubes inside AC tube bundle (inch)
AC_Nr = 5           # Number of tube rows in the bundle

#Pressure Vessel
Vessel_thickness = 0.25 # minimum Vessel Thickness for D < 4ft
Shell_density = 0.284 # lb/in3 shell density 
Resid_time = 0.5 # Liquied residence time in vessel (hr)

a_react = equipment_size_ref['Reactor']['a']
b_react = equipment_size_ref['Reactor']['b']
n_react = equipment_size_ref['Reactor']['n']
Sl_react = equipment_size_ref['Reactor']['S_lower']
Su_react = equipment_size_ref['Reactor']['S_upper']

a_Boiler = equipment_size_ref['HEX_Shell&Tube']['a']
b_Boiler = equipment_size_ref['HEX_Shell&Tube']['b']
n_Boiler = equipment_size_ref['HEX_Shell&Tube']['n']
Sl_Boiler = equipment_size_ref['HEX_Shell&Tube']['S_lower']
Su_Boiler = equipment_size_ref['HEX_Shell&Tube']['S_upper']

a_furnace = equipment_size_ref['Furnace_Cylindrical']['a']
b_furnace = equipment_size_ref['Furnace_Cylindrical']['b']
n_furnace = equipment_size_ref['Furnace_Cylindrical']['n']
Sl_furnace = equipment_size_ref['Furnace_Cylindrical']['S_lower']
Su_furnace = equipment_size_ref['Furnace_Cylindrical']['S_upper']

a_Aircooler = equipment_size_ref['Air-Cooler']['a']
b_Aircooler = equipment_size_ref['Air-Cooler']['b']
n_Aircooler = equipment_size_ref['Air-Cooler']['n']
Sl_Aircooler = equipment_size_ref['Air-Cooler']['S_lower']
Su_Aircooler = equipment_size_ref['Air-Cooler']['S_upper']

a_Vessel = equipment_size_ref['Vertical_Vessel']['a']
b_Vessel = equipment_size_ref['Vertical_Vessel']['b']
n_Vessel = equipment_size_ref['Vertical_Vessel']['n']
Sl_Vessel = equipment_size_ref['Vertical_Vessel']['S_lower']
Su_Vessel = equipment_size_ref['Vertical_Vessel']['S_upper']

# ********************** ENTHALPY CALCULATIONS ****************************** #

# Steam table (data/steam_table.csv, exported from the 'Steam_Table' sheet of
# data/project_database.xlsx). The plant steam levels are table entries, so the
# lookups below return the tabulated values exactly; between entries they
# interpolate linearly.
FILE = os.path.join(_HERE, '..', 'data', 'steam_table.csv')
if not os.path.exists(FILE):
    raise FileNotFoundError(FILE)

_steam = np.genfromtxt(FILE, delimiter=',', names=True)


# Enthalpy finder
def h_fg_calculator(T_sat):
    return float(np.interp(T_sat, _steam['T_sat'], _steam['h_fg']))


def h_g_calculator(T_sat):
    return float(np.interp(T_sat, _steam['T_sat'], _steam['h_g']))

h_fg_LP = h_fg_calculator(T_LP)
h_fg_MP = h_fg_calculator(T_MP)
h_fg_HP = h_fg_calculator(T_HP)

h_g_LP = h_g_calculator(T_LP)
h_g_MP = h_g_calculator(T_MP)
h_g_HP = h_g_calculator(T_HP)

# Price/kg of HP steam
def Steam_Price(p_fuel, p_bfw, h_fg):
    p_hps = ((p_fuel * h_fg)/n_boiler) + p_bfw
    return p_hps

p_hps = Steam_Price(p_fuel, p_bfw, h_fg_HP)
p_mps = p_hps - ((h_g_HP - h_g_MP) * n_turb * W_cost)
p_lps = p_mps - ((h_g_MP - h_g_LP) * n_turb * W_cost) # $/kmol 



interest_rate = 0.000000000001 # interest rate can be deleted in DAC
plant_life = 20             # years
operating_hours = 330*24

CEPCI_2023 = 900
CEPCI_2013 = 567
CEPCI_2007 = 509.7


#annualcapital charge ratio function, inputing i= interest rate and n= plant life
def annualized_capital(i,n):
    ACCR = (i * (1 + i)**n ) / ( (1 + i)**n - 1)
    return ACCR

ACCR = annualized_capital(interest_rate,plant_life)

#********************* External Black-box ************************************
# ------------------- TRF Black-box Function -------------------
def TRF_blackbox(inputs):
    # Unpack all inputs as floats
    t_ads, Q_air, x_H2, x_CO2, x_CO2_m, D, L, Q_DFM, DP_fan3 = [float(x) for x in inputs]

    # --- Adsorption ---
    q_dfm = q_dfm_eq - q_dfm_eq * math.exp(-k_ads * t_ads)

    # --- CO2 captured & reactor weight ---
    DFM_Weight = (x_CO2 * t_ads) / (q_dfm * 3600)

    # --- Reactor size ---
    S_react = (math.pi * D**2 / 4) * L

    # --- Number of reactors ---
    N_react = DFM_Weight / (S_react * 0.95 * DFM_SPWeight)

    # --- Heat of reaction ---
    Q_reaction = x_CO2_m * H_methanation / 3.6

    # --- Velocity & pressure drop ---
    v = (Q_air / N_react) / (math.pi * D**2 / 4)
    DP = 8 * L * air_Mu * v / Ri**2
    DP = DP_fan3
    
    # Costing
    Ce_react = (a_react + b_react*(S_react**n_react)) * N_react * CEPCI_2023/CEPCI_2007
    CAPEX_DFM = 2*(Ce_react + (DFM_unit_cost * DFM_Weight))
    OPEX_DFM = Q_DFM * kWh_cost

    return [float(DFM_Weight), float(S_react), float(N_react),
            float(q_dfm), float(Q_reaction), float(v), float(DP), float(CAPEX_DFM), float(OPEX_DFM), float(Ce_react)]


# ------------------- Finite-Difference Gradient -------------------
def grad_TRF_blackbox(inputs, fixed=None, h=1e-6):
    base = TRF_blackbox(inputs)
    n_outputs = len(base)
    n_inputs = len(inputs)
    grads = []
    for i in range(n_outputs):
        row = []
        for j in range(n_inputs):
            x_plus = list(inputs[:])
            x_plus[j] += h
            f_plus = TRF_blackbox(x_plus)[i]
            row.append(float((f_plus - base[i]) / h))
        grads.append(row)
    return grads



# ********************* OPTIMISATION MODEL ************************************
m = ConcreteModel()

# Index Sets
m.K = Set(initialize=Streams, ordered=False)
m.C = Set(initialize=Components.keys())


# Decision Variables - Mass & Energy Streams

x_init_values = {
    (1, 'CH4'): 0, (1, 'CO2'): 26.0, (1, 'H2'): 0, (1, 'H2O'): 0, (1, 'N2'): 51226.0, (1, 'O2'): 13747.0,
    (2, 'CH4'): 0, (2, 'CO2'): 0, (2, 'H2'): 0, (2, 'H2O'): 104.0, (2, 'N2'): 0, (2, 'O2'): 0,
    (3, 'CH4'): None, (3, 'CO2'): None, (3, 'H2'): None, (3, 'H2O'): None, (3, 'N2'): None, (3, 'O2'): None,
    (4, 'CH4'): None, (4, 'CO2'): None, (4, 'H2'): None, (4, 'H2O'): None, (4, 'N2'): None, (4, 'O2'): None,
    (5, 'CH4'): 0, (5, 'CO2'): 0, (5, 'H2'): 104.0, (5, 'H2O'): 0, (5, 'N2'): 0, (5, 'O2'): 0,
    (6, 'CH4'): 0, (6, 'CO2'): 26.0, (6, 'H2'): 0, (6, 'H2O'): 0, (6, 'N2'): 51226.0, (6, 'O2'): 13747.0,
    (7, 'CH4'): 25.99999999312702, (7, 'CO2'): 0, (7, 'H2'): 0, (7, 'H2O'): 51.99999999812516, (7, 'N2'): 0, (7, 'O2'): 0,
    (8, 'CH4'): 25.99999999312702, (8, 'CO2'): 0, (8, 'H2'): 0, (8, 'H2O'): 51.99999999906258, (8, 'N2'): 0, (8, 'O2'): 0,
    (9, 'CH4'): 25.99999999312702, (9, 'CO2'): 0, (9, 'H2'): 0, (9, 'H2O'): 52, (9, 'N2'): 0, (9, 'O2'): 0,
    (10, 'CH4'): 25.99999999312702, (10, 'CO2'): 0, (10, 'H2'): 0, (10, 'H2O'): 0, (10, 'N2'): 0, (10, 'O2'): 0,
    (11, 'CH4'): None, (11, 'CO2'): None, (11, 'H2'): None, (11, 'H2O'): None, (11, 'N2'): None, (11, 'O2'): None,
    (12, 'CH4'): None, (12, 'CO2'): None, (12, 'H2'): None, (12, 'H2O'): None, (12, 'N2'): None, (12, 'O2'): None,
    (13, 'CH4'): None, (13, 'CO2'): None, (13, 'H2'): None, (13, 'H2O'): None, (13, 'N2'): None, (13, 'O2'): None,
    (14, 'CH4'): None, (14, 'CO2'): None, (14, 'H2'): None, (14, 'H2O'): None, (14, 'N2'): None, (14, 'O2'): None,
    (15, 'CH4'): 0, (15, 'CO2'): 26.0, (15, 'H2'): 0, (15, 'H2O'): 0, (15, 'N2'): 51226.0, (15, 'O2'): 13747.0,
    (16, 'CH4'): 0, (16, 'CO2'): 26.0, (16, 'H2'): 0, (16, 'H2O'): 0, (16, 'N2'): 51226.0, (16, 'O2'): 13747.0,
    (20, 'CH4'): 0, (20, 'CO2'): 0, (20, 'H2'): 0, (20, 'H2O'): 52.0, (20, 'N2'): 0, (20, 'O2'): 0,
    (21, 'CH4'): None, (21, 'CO2'): None, (21, 'H2'): None, (21, 'H2O'): None, (21, 'N2'): None, (21, 'O2'): None,
    (22, 'CH4'): None, (22, 'CO2'): None, (22, 'H2'): None, (22, 'H2O'): None, (22, 'N2'): None, (22, 'O2'): None,
    (23, 'CH4'): 0, (23, 'CO2'): 0, (23, 'H2'): 0, (23, 'H2O'): 104.0, (23, 'N2'): 0, (23, 'O2'): 0,
    (24, 'CH4'): 0, (24, 'CO2'): 0, (24, 'H2'): 104.0, (24, 'H2O'): 0, (24, 'N2'): 0, (24, 'O2'): 0,
    (25, 'CH4'): None, (25, 'CO2'): None, (25, 'H2'): None, (25, 'H2O'): None, (25, 'N2'): None, (25, 'O2'): None,
    (26, 'CH4'): None, (26, 'CO2'): None, (26, 'H2'): None, (26, 'H2O'): None, (26, 'N2'): None, (26, 'O2'): None,
    (27, 'CH4'): None, (27, 'CO2'): None, (27, 'H2'): None, (27, 'H2O'): None, (27, 'N2'): None, (27, 'O2'): None,
    (28, 'CH4'): None, (28, 'CO2'): None, (28, 'H2'): None, (28, 'H2O'): None, (28, 'N2'): None, (28, 'O2'): None,
    (29, 'CH4'): 0, (29, 'CO2'): 0, (29, 'H2'): 0, (29, 'H2O'): 0, (29, 'N2'): 0, (29, 'O2'): 52.00000000124829,
    (51, 'CH4'): 0, (51, 'CO2'): 0, (51, 'H2'): 104.0, (51, 'H2O'): 0, (51, 'N2'): 0, (51, 'O2'): 0,
    (52, 'CH4'): None, (52, 'CO2'): None, (52, 'H2'): None, (52, 'H2O'): None, (52, 'N2'): None, (52, 'O2'): None,
    (61, 'CH4'): 0, (61, 'CO2'): 26.0, (61, 'H2'): 0, (61, 'H2O'): 0, (61, 'N2'): 51226.0, (61, 'O2'): 13747.0,
    (62, 'CH4'): 0, (62, 'CO2'): 0, (62, 'H2'): 0, (62, 'H2O'): 0, (62, 'N2'): 51226.0, (62, 'O2'): 13747.0,
    (63, 'CH4'): 0, (63, 'CO2'): 26.0, (63, 'H2'): 0, (63, 'H2O'): 0, (63, 'N2'): 0, (63, 'O2'): 0,
    (64, 'CH4'): 50000.0, (64, 'CO2'): 50000.0, (64, 'H2'): 50000.0, (64, 'H2O'): 50000.0, (64, 'N2'): 50000.0, (64, 'O2'): 50000.0,
    (65, 'CH4'): 50000.0, (65, 'CO2'): 0, (65, 'H2'): 50000.0, (65, 'H2O'): 50000.0, (65, 'N2'): 50000.0, (65, 'O2'): 50000.0,
    (66, 'CH4'): 0, (66, 'CO2'): 50000.0, (66, 'H2'): 0, (66, 'H2O'): 0, (66, 'N2'): 0, (66, 'O2'): 0,
    (91, 'CH4'): 0, (91, 'CO2'): 0, (91, 'H2'): 0, (91, 'H2O'): 15.756373405750006, (91, 'N2'): 0, (91, 'O2'): 0,
    (92, 'CH4'): 0, (92, 'CO2'): 0, (92, 'H2'): 0, (92, 'H2O'): None, (92, 'N2'): 0, (92, 'O2'): 0, # you may initialize none to zero in 92 and 93
    (93, 'CH4'): None, (93, 'CO2'): 0, (93, 'H2'): 0, (93, 'H2O'): None, (93, 'N2'): 0, (93, 'O2'): 0,
    (94, 'CH4'): 0, (94, 'CO2'): 0, (94, 'H2'): 0, (94, 'H2O'): 15.756373405750006, (94, 'N2'): 0, (94, 'O2'): 0,
    (95, 'CH4'): None, (95, 'CO2'): None, (95, 'H2'): None, (95, 'H2O'): None, (95, 'N2'): None, (95, 'O2'): None,
    (96, 'CH4'): None, (96, 'CO2'): None, (96, 'H2'): None, (96, 'H2O'): None, (96, 'N2'): None, (96, 'O2'): None,
    (635, 'CH4'): 0, (635, 'CO2'): 26.000000003124125, (635, 'H2'): 104.00000000249669, (635, 'H2O'): 0, (635, 'N2'): 0, (635, 'O2'): 0,
    (636, 'CH4'): 25.99999999312702, (636, 'CO2'): 0, (636, 'H2'): 0, (636, 'H2O'): 51.99999999718774, (636, 'N2'): 0, (636, 'O2'): 0,
    (665, 'CH4'): None, (665, 'CO2'): None, (665, 'H2'): None, (665, 'H2O'): None, (665, 'N2'): None, (665, 'O2'): None,
    (666, 'CH4'): None, (666, 'CO2'): None, (666, 'H2'): None, (666, 'H2O'): None, (666, 'N2'): None, (666, 'O2'): None
}

# x_init_values = {
#     (1, 'CH4'): 0, (1, 'CO2'): 26.0, (1, 'H2'): 0, (1, 'H2O'): 0, (1, 'N2'): 51226.0, (1, 'O2'): 13747.0,
#     (2, 'CH4'): 0, (2, 'CO2'): 0, (2, 'H2'): 0, (2, 'H2O'): 104.0, (2, 'N2'): 0, (2, 'O2'): 0,
#     (3, 'CH4'): 1e-10, (3, 'CO2'): 1e-10, (3, 'H2'): 1e-10, (3, 'H2O'): 1e-10, (3, 'N2'): 1e-10, (3, 'O2'): 1e-10,
#     (4, 'CH4'): 1e-10, (4, 'CO2'): 1e-10, (4, 'H2'): 1e-10, (4, 'H2O'): 1e-10, (4, 'N2'): 1e-10, (4, 'O2'): 1e-10,
#     (5, 'CH4'): 0, (5, 'CO2'): 0, (5, 'H2'): 104.0, (5, 'H2O'): 0, (5, 'N2'): 0, (5, 'O2'): 0,
#     (6, 'CH4'): 0, (6, 'CO2'): 26.0, (6, 'H2'): 0, (6, 'H2O'): 0, (6, 'N2'): 51226.0, (6, 'O2'): 13747.0,
#     (7, 'CH4'): 25.99999999312702, (7, 'CO2'): 0, (7, 'H2'): 0, (7, 'H2O'): 51.99999999812516, (7, 'N2'): 0, (7, 'O2'): 0,
#     (8, 'CH4'): 25.99999999312702, (8, 'CO2'): 0, (8, 'H2'): 0, (8, 'H2O'): 51.99999999906258, (8, 'N2'): 0, (8, 'O2'): 0,
#     (9, 'CH4'): 25.99999999312702, (9, 'CO2'): 0, (9, 'H2'): 0, (9, 'H2O'): 52, (9, 'N2'): 0, (9, 'O2'): 0,
#     (10, 'CH4'): 25.99999999312702, (10, 'CO2'): 0, (10, 'H2'): 0, (10, 'H2O'): 0, (10, 'N2'): 0, (10, 'O2'): 0,
#     (11, 'CH4'): 1e-10, (11, 'CO2'): 1e-10, (11, 'H2'): 1e-10, (11, 'H2O'): 1e-10, (11, 'N2'): 1e-10, (11, 'O2'): 1e-10,
#     (12, 'CH4'): 1e-10, (12, 'CO2'): 1e-10, (12, 'H2'): 1e-10, (12, 'H2O'): 1e-10, (12, 'N2'): 1e-10, (12, 'O2'): 1e-10,
#     (13, 'CH4'): 1e-10, (13, 'CO2'): 1e-10, (13, 'H2'): 1e-10, (13, 'H2O'): 1e-10, (13, 'N2'): 1e-10, (13, 'O2'): 1e-10,
#     (14, 'CH4'): 1e-10, (14, 'CO2'): 1e-10, (14, 'H2'): 1e-10, (14, 'H2O'): 1e-10, (14, 'N2'): 1e-10, (14, 'O2'): 1e-10,
#     (15, 'CH4'): 0, (15, 'CO2'): 26.0, (15, 'H2'): 0, (15, 'H2O'): 0, (15, 'N2'): 51226.0, (15, 'O2'): 13747.0,
#     (16, 'CH4'): 0, (16, 'CO2'): 26.0, (16, 'H2'): 0, (16, 'H2O'): 0, (16, 'N2'): 51226.0, (16, 'O2'): 13747.0,
#     (20, 'CH4'): 0, (20, 'CO2'): 0, (20, 'H2'): 0, (20, 'H2O'): 52.0, (20, 'N2'): 0, (20, 'O2'): 0,
#     (21, 'CH4'): 1e-10, (21, 'CO2'): 1e-10, (21, 'H2'): 1e-10, (21, 'H2O'): 1e-10, (21, 'N2'): 1e-10, (21, 'O2'): 1e-10,
#     (22, 'CH4'): 1e-10, (22, 'CO2'): 1e-10, (22, 'H2'): 1e-10, (22, 'H2O'): 1e-10, (22, 'N2'): 1e-10, (22, 'O2'): 1e-10,
#     (23, 'CH4'): 0, (23, 'CO2'): 0, (23, 'H2'): 0, (23, 'H2O'): 104.0, (23, 'N2'): 0, (23, 'O2'): 0,
#     (24, 'CH4'): 0, (24, 'CO2'): 0, (24, 'H2'): 104.0, (24, 'H2O'): 0, (24, 'N2'): 0, (24, 'O2'): 0,
#     (25, 'CH4'): 1e-10, (25, 'CO2'): 1e-10, (25, 'H2'): 1e-10, (25, 'H2O'): 1e-10, (25, 'N2'): 1e-10, (25, 'O2'): 1e-10,
#     (26, 'CH4'): 1e-10, (26, 'CO2'): 1e-10, (26, 'H2'): 1e-10, (26, 'H2O'): 1e-10, (26, 'N2'): 1e-10, (26, 'O2'): 1e-10,
#     (27, 'CH4'): 1e-10, (27, 'CO2'): 1e-10, (27, 'H2'): 1e-10, (27, 'H2O'): 1e-10, (27, 'N2'): 1e-10, (27, 'O2'): 1e-10,
#     (28, 'CH4'): 1e-10, (28, 'CO2'): 1e-10, (28, 'H2'): 1e-10, (28, 'H2O'): 1e-10, (28, 'N2'): 1e-10, (28, 'O2'): 1e-10,
#     (29, 'CH4'): 0, (29, 'CO2'): 0, (29, 'H2'): 0, (29, 'H2O'): 0, (29, 'N2'): 0, (29, 'O2'): 52.00000000124829,
#     (51, 'CH4'): 0, (51, 'CO2'): 0, (51, 'H2'): 104.0, (51, 'H2O'): 0, (51, 'N2'): 0, (51, 'O2'): 0,
#     (52, 'CH4'): 1e-10, (52, 'CO2'): 1e-10, (52, 'H2'): 1e-10, (52, 'H2O'): 1e-10, (52, 'N2'): 1e-10, (52, 'O2'): 1e-10,
#     (61, 'CH4'): 0, (61, 'CO2'): 26.0, (61, 'H2'): 0, (61, 'H2O'): 0, (61, 'N2'): 51226.0, (61, 'O2'): 13747.0,
#     (62, 'CH4'): 0, (62, 'CO2'): 0, (62, 'H2'): 0, (62, 'H2O'): 0, (62, 'N2'): 51226.0, (62, 'O2'): 13747.0,
#     (63, 'CH4'): 0, (63, 'CO2'): 26.0, (63, 'H2'): 0, (63, 'H2O'): 0, (63, 'N2'): 0, (63, 'O2'): 0,
#     (64, 'CH4'): 50000.0, (64, 'CO2'): 50000.0, (64, 'H2'): 50000.0, (64, 'H2O'): 50000.0, (64, 'N2'): 50000.0, (64, 'O2'): 50000.0,
#     (65, 'CH4'): 50000.0, (65, 'CO2'): 0, (65, 'H2'): 50000.0, (65, 'H2O'): 50000.0, (65, 'N2'): 50000.0, (65, 'O2'): 50000.0,
#     (66, 'CH4'): 0, (66, 'CO2'): 50000.0, (66, 'H2'): 0, (66, 'H2O'): 0, (66, 'N2'): 0, (66, 'O2'): 0,
#     (91, 'CH4'): 0, (91, 'CO2'): 0, (91, 'H2'): 0, (91, 'H2O'): 15.756373405750006, (91, 'N2'): 0, (91, 'O2'): 0,
#     (92, 'CH4'): 0, (92, 'CO2'): 0, (92, 'H2'): 0, (92, 'H2O'): 1e-10, (92, 'N2'): 0, (92, 'O2'): 0, # you may initialize 1e-10 to zero in 92 and 93
#     (93, 'CH4'): 1e-10, (93, 'CO2'): 0, (93, 'H2'): 0, (93, 'H2O'): 1e-10, (93, 'N2'): 0, (93, 'O2'): 0,
#     (94, 'CH4'): 0, (94, 'CO2'): 0, (94, 'H2'): 0, (94, 'H2O'): 15.756373405750006, (94, 'N2'): 0, (94, 'O2'): 0,
#     (95, 'CH4'): 1e-10, (95, 'CO2'): 1e-10, (95, 'H2'): 1e-10, (95, 'H2O'): 1e-10, (95, 'N2'): 1e-10, (95, 'O2'): 1e-10,
#     (96, 'CH4'): 1e-10, (96, 'CO2'): 1e-10, (96, 'H2'): 1e-10, (96, 'H2O'): 1e-10, (96, 'N2'): 1e-10, (96, 'O2'): 1e-10,
#     (635, 'CH4'): 0, (635, 'CO2'): 26.000000003124125, (635, 'H2'): 104.00000000249669, (635, 'H2O'): 0, (635, 'N2'): 0, (635, 'O2'): 0,
#     (636, 'CH4'): 25.99999999312702, (636, 'CO2'): 0, (636, 'H2'): 0, (636, 'H2O'): 51.99999999718774, (636, 'N2'): 0, (636, 'O2'): 0,
#     (665, 'CH4'): 1e-10, (665, 'CO2'): 1e-10, (665, 'H2'): 1e-10, (665, 'H2O'): 1e-10, (665, 'N2'): 1e-10, (665, 'O2'): 1e-10,
#     (666, 'CH4'): 1e-10, (666, 'CO2'): 1e-10, (666, 'H2'): 1e-10, (666, 'H2O'): 1e-10, (666, 'N2'): 1e-10, (666, 'O2'): 1e-10
# }



m.x = Var(m.K, m.C, domain=NonNegativeReals, initialize=x_init_values, bounds=(0,1e6))    # kmol/hr

m.x[9,'H2O'].fix(52)

T_init_values = {
    1: None, 2: None, 3: None, 4: None, 5: 700.0, 6: None, 7: 350.0, 8: 169.9999999000718,
    9: None, 10: None, 11: None, 12: None, 13: None, 14: None, 15: None, 16: None,
    20: None, 21: None, 22: None, 23: None, 24: 700.0, 25: None, 26: None, 27: None,
    28: None, 29: None, 51: None, 52: None, 61: None, 62: None, 63: None, 64: None,
    65: None, 66: None, 91: None, 92: None, 93: None, 94: None, 95: None, 96: None,
    635: None, 636: 350.0, 665: None, 666: None
}


# T_init_values = {
#     1: 1e-10, 2: 1e-10, 3: 1e-10, 4: 1e-10, 5: 700.0, 6: 1e-10, 7: 350.0, 8: 169.9999999000718,
#     9: 1e-10, 10: 1e-10, 11: 1e-10, 12: 1e-10, 13: 1e-10, 14: 1e-10, 15: 1e-10, 16: 1e-10,
#     20: 1e-10, 21: 1e-10, 22: 1e-10, 23: 1e-10, 24: 700.0, 25: 1e-10, 26: 1e-10, 27: 1e-10,
#     28: 1e-10, 29: 1e-10, 51: 1e-10, 52: 1e-10, 61: 1e-10, 62: 1e-10, 63: 1e-10, 64: 1e-10,
#     65: 1e-10, 66: 1e-10, 91: 1e-10, 92: 1e-10, 93: 1e-10, 94: 1e-10, 95: 1e-10, 96: 1e-10,
#     635: 1e-10, 636: 350.0, 665: 1e-10, 666: 1e-10
# }

# Decision Variables - Design
m.T = Var(m.K, domain = NonNegativeReals, initialize=T_init_values, bounds=(0,800)) # for stream 8: Set 170 for LP, 220 for MP and 270 for HP
m.T[24].fix(SOEL_T)
m.T[636].fix(350)

m.T_sat = Param(initialize=150, mutable=False) #Set 150 for LP, 200 for MP and 250 for HP

m.T_air_Out = Var(domain=NonNegativeReals, initialize=40, bounds=(35,40))      # Air-Cooler Outlet Air Temperature °C       #||||||||||||||||||||||||||||||||||||

m.LMTD_AC = Var(domain=NonNegativeReals, initialize=53.25345622742183)  

# m.v = Var(domain=NonNegativeReals, bounds=(0.1,10), initialize=2.199827623651915)         # air average velocity inside monolith channels   

m.Q_air = Var(domain=NonNegativeReals, initialize=434.1361055086352)

m.N_fan3 = Var(domain=NonNegativeReals, bounds=(1,20), initialize=6.008531239974121)             # Number of fans in system 3           #||||||||||||||||||||||||||||||||||||
m.N_fan = Var(domain=NonNegativeReals, bounds=(1,100), initialize=6.008531239974121)            # Total Number of fans                 #||||||||||||||||||||||||||||||||||||
m.Q_fan3 = Var(domain=NonNegativeReals, initialize=434.1361055086352)         # fans system 3 capacity
m.Q_fan3e = Var(domain=NonNegativeReals, bounds=(1e-5,None), initialize=72.253282569353)        # fans system 3 each fan capacity
m.DP_fan3 = Var(bounds=(522.5589435417271,2000), initialize=522.5589435417271)    # Fans produced head (pa)
m.FH3 = Param(initialize=1.15, mutable=False)
m.power_fan3 = Var(domain=NonNegativeReals, initialize=420.1142678665037)                                         # fans system 3 each fan power


# m.N_react = Var(domain=NonNegativeReals, initialize=15.71258410504647-0.014284167368241896+0.006318556851775625, bounds=(1,50))           # Number of reactors (Min: )           #|||||||||||||||||||||||||||||||||||| ##########
m.D = Var(domain=NonNegativeReals, initialize=4, bounds=(0.5,4))                    # DFM reactor Diameter                  #||||||||||||||||||||||||||||||||||||
m.L = Var(domain=NonNegativeReals, initialize=1.769552762755285, bounds=(1,5))                      # Reactor height/length                 #||||||||||||||||||||||||||||||||||||
m.DP = Var(domain=NonNegativeReals, bounds=(0,522.5589435417271), initialize=522.5589435417297)             # reactor pressure drop (pa)

# m.S_react = Var(domain=NonNegativeReals, bounds=(Sl_react,Su_react), initialize=22.22558270020638+0.02022346014577181-0.008950321906052494)                # Size of each reactor (m3)

# m.DFM_Weight = Var(domain=NonNegativeReals, initialize=28342.27991630082)                                         # Total Weight of DFM
# m.DFM_Weight.fix(28342.27991630082)

m.Q_DFM = Var(domain=NonNegativeReals, initialize=0)          # kWh external heat required
# m.q_dfm = Var(domain=NonNegativeReals, bounds=(0,0.99*q_dfm_eq), initialize=0.000254821497902)                    # kmole of CO2 / kg DFM
m.t_ads = Var(initialize=1000, bounds=(500,3000))         # adsorption time             # ||||||||||||||||||||||||||||||||||||||||||||###
# m.t_ads.fix(1000)

m.Q_H2 = Var(domain=NonNegativeReals, initialize=577.0)           # Heat supplied via Hydrogen stream
# m.Q_reaction = Var(domain=NonNegativeReals, initialize=1184.444444444446)     # Heat of reaction

m.Q_ex = Var(domain=NonNegativeReals, initialize=599975.9999999994+0.0002539907582104206)
m.Q_LP = Var(domain=NonNegativeReals, initialize=599975.9999999994+0.0002539907582104206)

m.A_Boiler = Var(domain=NonNegativeReals, bounds=(Sl_Boiler,Su_Boiler), initialize=65.04217487543131)             # Size of the Heat Exchanger (m2)
m.N_Boiler = Var(domain=NonNegativeReals, bounds=(1,10), initialize=1)                         # Number of Boilers/HEX

#m.A_Aircooler = Var(domain=NonNegativeReals, bounds=(Sl_Aircooler,Su_Aircooler))   # Required Area of the Air-Cooler (ft2)
m.A_Aircooler = Var(bounds=(1e-4,None), initialize=185.7260957290091)                                        # Required Area of the Air-Cooler (ft2)
m.BHP_Aircooler = Var(domain=NonNegativeReals, initialize=6.214676344014418)                                      # Air-Cooler power requirement (horse power)
m.Q_Aircooler = Var(domain=NonNegativeReals, initialize=2547066.392000001-0.0004237731918692589)    # Heat Duty of Air-Cooler

m.Z_AC = Var(domain=NonNegativeReals, initialize=0.896551724137931)                                               # Dimensionless AC parameter
m.Y_AC = Var(domain=NonNegativeReals, initialize=2.084331296110056)  

m.F_A = Var(domain=NonNegativeReals, initialize=83.3732518444022)                                                # Face area of Air-Cooler ft^2
m.F_V = Var(domain=NonNegativeReals, bounds=(400,650), initialize=550)                              # Std. Air Face Velocity ft/min
m.A_AC_av = Var(domain=NonNegativeReals, initialize=523.5840215828459)                                            # Available heat transfer Area of the Air-Cooler (ft2)

m.DP_AC = Var(domain=NonNegativeReals, initialize=0.397946390156841)          # Air-Cooler air-side pressure drop (inch of water)

m.S_Vessel = Var(domain=NonNegativeReals, bounds=(Sl_Vessel,Su_Vessel), initialize=338.8632396196883)             # Total Size of contactors (m3)
m.D_Vessel = Var(domain=NonNegativeReals, initialize=0.668008620353308)                                           # Vessel Diameter, m
m.L_Vessel = Var(domain=NonNegativeReals, initialize=2.672034481413233)   

# Decision Variables - Economy

m.CAPEX = Var(domain=NonNegativeReals, initialize=3197019.462769468-302.5435943841003)
m.OPEX = Var(domain=NonNegativeReals, initialize=3283938.870643495)

m.OPEX_SOEL = Var(domain=NonNegativeReals, initialize=3115738.414131202)
m.CAPEX_SOEL = Var(domain=NonNegativeReals, initialize=32747866.6666666)

m.Ce_fan3 = Var(domain=NonNegativeReals, initialize=444168.4278744024)                     # purchased equipment cost, 2013 basis
m.Ce_fan3_update = Var(domain=NonNegativeReals, initialize=705029.2505942909)              # purchased equipment cost, 2023 basis
m.CAPEX_FAN = Var(domain=NonNegativeReals, initialize=705029.2505942909)
m.OPEX_FAN = Var(domain=NonNegativeReals, initialize=166365.2500751354)

# m.Ce_react = Var(domain=NonNegativeReals, initialize=10756183.62954394-3025.704906763509+1338.562692988664)                    # purchased Reactor cost, 2007 basis
# m.CAPEX_DFM = Var(domain=NonNegativeReals, initialize=30366495.50494025-6051.409813515842+2677.1253859698772)
# m.OPEX_DFM = Var(domain=NonNegativeReals, initialize=0)

m.Ce_Boiler = Var(domain=NonNegativeReals, initialize=30895.91496329419)                   # purchased equipment cost, 2007 basis
m.CAPEX_Boiler = Var(domain=NonNegativeReals, initialize=54554.29363736455)

# m.Ce_Furnace = Var(domain=NonNegativeReals, initialize=101609.231742407)                  # purchased Furnace cost, 2007 basis
# m.CAPEX_Furnace = Var(domain=NonNegativeReals, initialize=179415.9477499826)
# m.OPEX_Furnace = Var(domain=NonNegativeReals, initialize=108900.0)

m.Ce_Aircooler = Var(domain=NonNegativeReals, initialize=29754.0358115308)                # purchased Furnace cost, 2013 basis
m.CAPEX_Aircooler = Var(domain=NonNegativeReals, initialize=47228.6282722712)
m.OPEX_Aircooler = Var(domain=NonNegativeReals, initialize=1835.206437158621)

m.Ce_PVessel = Var(domain=NonNegativeReals, initialize=14101.27402637812)                  # purchased Pressure vessel cost, 2007 basis
m.CAPEX_PVessel = Var(domain=NonNegativeReals, initialize=24899.24783939629)

m.CAPEX_Separator = Var(domain=NonNegativeReals, initialize=0)

m.Profit = Var(domain=NonNegativeReals, initialize=1879229.142722491-0.00048478930330020376)
m.Profit_Product = Var(domain=NonNegativeReals, initialize=1853279.99999991-0.0004898160696029663)
m.Profit_Steam = Var(domain=NonNegativeReals, initialize=25949.14272258099+5.0268245104234666e-06)


# ************************ Registering External Functions ******************* #

# ------------------- ExternalFunction Registration -------------------
outputs = ["DFM_Weight", "S_react", "N_react", "q_dfm", "Q_reaction", "v", "DP", "CAPEX_DFM", "OPEX_DFM", "Ce_react"]
m.TRF_funcs = {}
m._ef_pycb_to_idx = {}  # map internal Pyomo callback to index

for idx, name in enumerate(outputs):
    # Safe closure for function
    def make_func(i):
        return lambda *args: float(TRF_blackbox(list(args))[i])
    # Safe closure for gradient
    def make_grad(i):
        return lambda args, fixed: list(grad_TRF_blackbox(list(args), fixed)[i])

    ef = ExternalFunction(function=make_func(idx), gradient=make_grad(idx))
    m.TRF_funcs[name] = ef
    m._ef_pycb_to_idx[ef._fcn] = idx

# ************************** CONSTRAINTS ************************************ #

# ********************* INITIAL & SEPARATION STATES ************************* #
@m.Constraint(m.C)
def Initial_Condition1(m,c):
    return (m.x[1,c] - Initial_Streams[1][c] == 0)

@m.Constraint(m.C)
def Initial_Condition2(m, c):
    if c == 'H2O':
        return Constraint.Skip   # 'H2' component should be variable
    else:
        return m.x[2, c] == 0.0  # set other components to zero

@m.Constraint(m.C)
def DFM_Adsorption1(m,c):
    if c == 'CO2':
        return m.x[62,c] == 0.0
    else:
        return m.x[62,c] == m.x[61,c]

@m.Constraint(m.C)
def DFM_Adsorption2(m,c): 
    if c == 'CO2':
        return m.x[63,c] == m.x[61,c]
    else:
        return m.x[63,c] == 0.0
    
@m.Constraint(m.C)
def VTSA_Adsorption1(m,c):
    if c == 'CO2':
        return m.x[65,c] == 0.0
    else:
        return m.x[65,c] == m.x[64,c]

@m.Constraint(m.C)
def VTSA_Adsorption2(m,c): 
    if c == 'CO2':
        return m.x[66,c] == m.x[64,c]
    else:
        return m.x[66,c] == 0.0

@m.Constraint(m.C)
def Steam_Composition(m, c):
    for i in (91, 92, 93):
        if c != 'H2O':
            return m.x[i,c] == 0.0
        else:
            continue # H2O remains as variable
    return Constraint.Feasible # return a valid expression for other components

@m.Constraint(m.C)
def Flash_Sparator1(m,c): 
    if c == 'H2O':
        return m.x[20,c] == m.x[9,c]
    else:
        return m.x[20,c] == 0.0

@m.Constraint(m.C)
def Flash_Sparator3(m,c): 
    if c == 'H2O':
        return m.x[10,c] == 0.0
    else:
        return m.x[10,c] == m.x[9,c]


# ***************************** NODES *************************************** #
# Electrolysers
m.constn01 = Constraint(m.C, rule = lambda m, c : m.x[2,c] ==  m.x[23,c])
m.constn02 = Constraint(m.C, rule = lambda m, c : m.x[5,c] == m.x[24,c])
m.constn04 = Constraint(m.C, rule = lambda m, c : m.x[23,c] <= M)

m.constT01 = Constraint(m.C, rule = lambda m, c : m.T[5] == m.T[24])
m.constT03 = Constraint(m.C, rule = lambda m, c : m.T[24] == SOEL_T)

# Air-Intake
m.constn07 = Constraint(m.C, rule = lambda m, c : m.x[1,c] == m.x[15,c])
m.constn10 = Constraint(m.C, rule = lambda m, c : m.x[15,c] <= M)

m.constn14 = Constraint(m.C, rule = lambda m, c : m.x[16,c] == m.x[15,c])
m.constn15 = Constraint(m.C, rule = lambda m, c : m.x[6,c] == m.x[16,c])

# Adsorption
m.constn16 = Constraint(m.C, rule = lambda m, c : m.x[6,c] == m.x[61,c])
m.constn17 = Constraint(m.C, rule = lambda m, c : m.x[7,c] == m.x[636,c])
m.constn18 = Constraint(m.C, rule = lambda m, c : m.x[61,c] <= M)

# DFM Mixer
m.constn21 = Constraint(m.C, rule = lambda m, c : m.x[635,c] == m.x[63,c] + m.x[51,c])

m.constn23 = Constraint(m.C, rule = lambda m, c : m.x[5,c] == m.x[51,c])
m.constn24 = Constraint(m.C, rule = lambda m, c : m.x[51,c] <= M)

# Steam Generation
m.constn26 = Constraint(expr = m.T[7] == m.T[636])
m.constn27 = Constraint(expr = m.T[636] == 350)     # this constraint shall be changed based on thermo-kinetics of reactor in future

m.constn29 = Constraint(m.C, rule = lambda m, c : m.x[7,c] == m.x[8,c])

m.constn30 = Constraint(expr = m.x[91,'H2O'] <= M)

m.constn34 = Constraint(m.C, rule = lambda m, c : m.x[94,c] == m.x[91,c])

#Air-Cooler
m.constn40 = Constraint(m.C, rule = lambda m, c : m.x[8,c] == m.x[9,c])

# ***************************** UNIT MODELS ********************************* #

# **************************** Electrolysers ********************************

# SOEL electrolyser model & pertaining Costs:

@m.Constraint(m.C)
def SOEL(m,c): 
    if c == 'H2':
        return m.x[24,c] - (m.x[23,c] + Stoich_ratio1[c] * Conv_fac1 * m.x[23,'H2O']) == 0
    else:
        return m.x[24,c] == 0.0
@m.Constraint(m.C)
def SOEL2(m,c): 
    if c == 'H2':
        return m.x[29,c] == 0
    else:
        return m.x[29,c] - (m.x[23,c] + Stoich_ratio1[c] * Conv_fac1 * m.x[23,'H2O']) == 0

#m.constSOEL01 = Constraint(m.C, rule = lambda m, c : m.x[24,c] - (m.x[23,c] + 
                                                                #Stoich_ratio1[c] * Conv_fac1 * m.x[23,'H2O']) == 0)
m.constSOEL02 = Constraint(expr = m.OPEX_SOEL == (m.x[24,'H2'] * W_cost * SOEL_unit_cost + 
                                                 m.x[23,'H2O'] * Water_cost)*operating_hours)
m.constSOEL03 = Constraint(expr = m.CAPEX_SOEL == (m.x[24,'H2'])* SOEL_unit_cost * SOEL_CAPEX/3600)


# **************************** Air-intake System ******************************

# Air-intake mass-balance model:

m.constAI03= Constraint(expr = m.Q_fan3 == sum(m.x[15,c]*Components[c]['MW']*
                                               (1/Components[c]['ro'])*(1/3600) for c in m.C)) # m3/s air flow to fans system 3

m.constAI06= Constraint(expr = m.Q_fan3e == m.Q_fan3 / m.N_fan3) # m3/s air flow to each fan of system 3

#Fan systems capacity limits
#m.constAI12= Constraint(expr = m.Q_fan3e >= Air_intake_units_Q_min) # min Capacity limit for fan of system 3
m.constAI13= Constraint(expr = m.Q_fan3e <= Air_intake_units_Q_max) # max Capacity limit for fan of system 3

#Fan systems pressure limits
m.constAI16= Constraint(expr = m.DP_fan3 <= Air_intake_units_Fan_maxH) # max head limit for system 3 fans


#Cost function for Centrifugal backward-curved fan (valid from Q = 1,000 to 100,000 ACFM): Seider et al 2016

m.constAI29= Constraint(expr = m.Ce_fan3 == (((8e-14*(m.Q_fan3e*2118.88)**3)+(2e-8*(m.Q_fan3e*2118.88)**2)+(0.1562*m.Q_fan3e*2118.88)+1042.9)* m.N_fan3 * m.FH3 * FM))
m.constAI30= Constraint(expr = m.Ce_fan3_update == m.Ce_fan3 * (CEPCI_2023/CEPCI_2013))

# Total fan powers
m.constAI33= Constraint(expr = m.power_fan3 == (m.Q_fan3e * m.DP_fan3 / (1e3*nf_fan*nm_fan )) * m.N_fan3) # m3/s x N/m2 x kJ/1e3J = kJ/s (kW)


# AI CAPEX & OPEX
m.constAI34= Constraint(expr= m.CAPEX_FAN == m.Ce_fan3_update)

m.constAI35= Constraint(expr= m.OPEX_FAN == m.power_fan3 * operating_hours * kWh_cost) #kW * hr * $/kWh = $

m.constAI36= Constraint(expr= m.N_fan == m.N_fan3)

# ********************************* DFM Reactor *******************************

# Air intake (Q_air comes from component flows)
m.constDFM01 = Constraint(expr = m.Q_air == sum(m.x[6,c]*Components[c]['MW']*(1/Components[c]['ro'])*(1/3600) for c in m.C))

# ------------------- TRF Inputs -------------------
TRF_inputs = [m.t_ads, m.Q_air, m.x[5,'H2'], m.x[6,'CO2'], m.x[63,'CO2'], m.D, m.L, m.Q_DFM, m.DP_fan3]

# ------------------- EF Constraints -------------------
# m.constTRF_blackbox = ConstraintList()
# for name in outputs:
#     var = getattr(m, name)
#     ef = m.TRF_funcs[name]
#     # Use EF directly
#     m.constTRF_blackbox.add(var == ef(*TRF_inputs))
    
# Vars to hold EF outputs
m.trf_out = Var(outputs, initialize=0)

# Constraints linking Vars to EF calls
def trf_constraints_rule(m, name):
    idx = outputs.index(name)
    return m.trf_out[name] == m.TRF_funcs[name](*TRF_inputs)

m.constTRF_blackbox = Constraint(outputs, rule=trf_constraints_rule)

# --- Other constraints (not in black-box) ---

# ------------------- Constraints using ExternalFunctions -------------------
# Fan pressure drop
m.constDFM05 = Constraint(expr = m.DP <= m.DP_fan3)

# Reactor stoichiometry (glass-box)
m.constDFM06 = Constraint(
    m.C,
    rule=lambda m, c: m.x[636,c] - (m.x[635,c] + Stoich_ratio2[c] * Conv_fac2 * m.x[635,'CO2']) == 0
)

# CAPEX / OPEX (black-box outputs)
# m.constDFM07 = Constraint(expr = m.CAPEX_DFM == 2*(m.Ce_react + (DFM_unit_cost * m.DFM_Weight)))
# m.constDFM08 = Constraint(expr = m.OPEX_DFM == m.Q_DFM * kWh_cost)
# m.constDFM09 = Constraint(expr = m.Ce_react == (a_react + b_react*(m.S_react**n_react)) * m.N_react * CEPCI_2023/CEPCI_2007)

# Energy use (glass-box)
m.constDFM17 = Constraint(expr = m.Q_H2 == m.x[5,'H2'] * 577/104)


# ***************************** Steam Generation ******************************
# m.constSTG01= Constraint(expr = m.T_sat == T_LP)
m.constSTG02= Constraint(expr = m.T[8] >= DT_min + m.T_sat)
m.constSTG04= Constraint(expr = m.Q_ex == sum(m.x[7,c] * Components[c]['cp'] for c in m.C)*(m.T[7] - m.T[8]))  #kJ/h
m.constSTG05= Constraint(expr = m.Q_ex == m.Q_LP)
m.constSTG06= Constraint(expr = m.Q_LP <= M)
m.constSTG09= Constraint(expr = m.x[91,'H2O'] == m.Q_LP / h_fg_LP)

# CAPEX for boiler
m.constHX02 = Constraint(expr = m.CAPEX_Boiler - (m.Ce_Boiler * (CEPCI_2023/CEPCI_2007)) == 0)
# Cost function for Boiler
m.constHX03 = Constraint(expr = m.Ce_Boiler == (a_Boiler + (b_Boiler * 
                                                            (m.A_Boiler ** n_Boiler)))*m.N_Boiler)  #Boiler cost equation, 2007 basis (to be checked)

# HEX Area
# m.constHX06 = Constraint(expr = m.T[7] - m.T[8] >= DT_min)  # e.g., DT_min = 20 K
m.constHX05= Constraint(expr = m.T[7] >= DT_min + m.T_sat)
log_term = log(m.T[7] - m.T_sat) - log(m.T[8] - m.T_sat) + 1e-9
m.constHX04 = Constraint(expr = m.A_Boiler == m.Q_ex / (U_Boiler * ((m.T[7] - m.T[8]) / log_term)))

# # Furnace
# m.constFRN01 = Constraint(expr = m.CAPEX_Furnace - (m.Ce_Furnace * (CEPCI_2023/CEPCI_2007)) == 0)
# m.constFRN02 = Constraint(expr = m.OPEX_Furnace == Q_Furnace * kWh_cost * operating_hours) 
# # Cost function for furnace
# m.constFRN03 = Constraint(expr = m.Ce_Furnace == (a_furnace + (b_furnace * 
#                                                             ((Q_Furnace/1000) ** n_furnace))))

# ******************************* Air-Cooler **********************************
# CAPEX for Air-Cooler
m.constAC01 = Constraint(expr = m.CAPEX_Aircooler - (m.Ce_Aircooler * (CEPCI_2023/CEPCI_2013)) == 0)
# Cost function for Air-Cooler
m.constAC02 = Constraint(expr = m.Ce_Aircooler == (a_Aircooler + (b_Aircooler * 
                                                            (m.A_Aircooler ** n_Aircooler))))  # AC cost equation, 2013 basis
# OPEX for Air-Cooler
m.constAC03 = Constraint(expr = m.OPEX_Aircooler == (m.BHP_Aircooler/1.341)*operating_hours*kWh_cost)

# Air-Cooler Area
#m.constAC04 = Constraint(expr = m.T_air_Out == 36.7) #in C
m.constAC05 = Constraint(expr = m.T[8] >= m.T_air_Out + DT_min)
m.constAC06 = Constraint(expr = m.LMTD_AC == ((m.T[8] - m.T_air_Out)-(T_product - T_air_In)) /(log((m.T[8] - m.T_air_Out)/(T_product - T_air_In)) + 1e-9)) # LMTD in C
m.constAC07 = Constraint(expr = m.A_Aircooler == (m.Q_Aircooler / ((U_Aircooler* 3.6 * m.LMTD_AC)+1e-9))*10.764)  # Area in ft2
m.constAC08 = Constraint(m.C, rule = lambda m, c : m.Q_Aircooler == sum(m.x[8,c]*Components[c]['cp'] for c in m.C)*(m.T[8]-T_product) + m.x[8,'H2O']*h_fg_calculator(100)) # Duty in kJ/hr
m.constAC09 = Constraint(expr = m.Z_AC == (m.T[8]-T_product)/(m.T[8]-T_air_In))
m.constAC10 = Constraint(expr = 0.7 - (m.Z_AC*100/(U_Aircooler/5.678)) <= 0.09)
m.constAC11 = Constraint(expr = 0.7 - (m.Z_AC*100/(U_Aircooler/5.678)) >= 0)
m.constAC13 = Constraint(expr = m.F_V == 550)  # F_V in ft/min
m.constAC14 = Constraint(expr = m.F_A == (m.Q_Aircooler/1.055)/((m.F_V *((m.T_air_Out*1.8+32)-(T_air_In*1.8+32))*1.95)+ 1e-9)) # F_A in ft2, Q in BTU/hr, Temp in F
m.constAC15 = Constraint(expr = m.Y_AC == m.F_A/L_AC_tube) # Bundle width (ft)
#m.constAC16 = Constraint(expr = (m.T_air_Out*1.8+32) == (T_air_In*1.8+32) + ((m.Q_Aircooler/1.055)/((m.Y_AC*m.F_V*L_AC_tube*1.95)+ 1e-9)))
m.constAC16 = Constraint(expr = m.A_AC_av == AC_Nr * (m.Y_AC/(AC_tube_gap/12))*3.14*(AC_tube_OD/12)*L_AC_tube) # Available heat transfer area (ft2)
m.constAC17 = Constraint(expr = m.A_AC_av >= m.A_Aircooler)

# Air-Cooler break horse power
m.constAC18 = Constraint(expr = m.BHP_Aircooler == (m.F_V*m.F_A*(m.T_air_Out + 273)*(m.DP_AC+0.1))/1.15e6)
m.constAC19 = Constraint(expr = m.DP_AC == 0.0037*AC_Nr*(m.F_V/100)**1.8)

# *********************** Separator (Pressure-Vessel) *************************
# CAPEX for Vertical Pressure Vessel
m.constPV01 = Constraint(expr = m.CAPEX_PVessel - (m.Ce_PVessel * (CEPCI_2023/CEPCI_2007)) == 0)
m.constPV02 = Constraint(expr = m.Ce_PVessel == (a_Vessel + (b_Vessel * 
                                                            (m.S_Vessel ** n_Vessel))))  # pressure vessel cost equation, 2007 basis
m.constPV03 = Constraint(expr = m.S_Vessel == 3.14*((m.D_Vessel*39.37) + Vessel_thickness)*((m.L_Vessel*39.37) + (0.8*m.D_Vessel*39.37))*Vessel_thickness*Shell_density/2.205) # kg Vessel
m.constPV04 = Constraint(expr = m.D_Vessel == (2*m.x[9,'H2O']*18*Resid_time/(1000*3.14))**(1/3))
m.constPV05 = Constraint(expr = m.L_Vessel == 4 * m.D_Vessel)

# ***************************** COST FUNCTIONS ********************************
# Annual Profit equation:
m.constpr01 = Constraint(expr = m.Profit - m.Profit_Product  - m.Profit_Steam <= 1e-6)
m.constpr02 = Constraint(expr = m.Profit_Product - (sum((m.x[7,c])*Components[c]['price'] for c in m.C))*operating_hours <= 1e-6)
m.constpr03 = Constraint(expr = m.Profit_Steam == ((m.x[91,'H2O'] * p_lps)) * operating_hours)

# # Annualized CAPEX equation:
m.constCX = Constraint(expr = m.CAPEX - ((m.CAPEX_FAN + m.trf_out["CAPEX_DFM"] + 
                                         m.CAPEX_SOEL + m.CAPEX_Boiler
                                         + m.CAPEX_Aircooler + m.CAPEX_Separator + m.CAPEX_PVessel) * ACCR) <= 1e-6)
# #m.constCX_ads = Constraint(expr = m.CAPEX_ads - (m.CAPEX_DFM) == 0)
m.constOX = Constraint(expr = m.OPEX - (m.OPEX_FAN + (m.trf_out["OPEX_DFM"]) 
                       + (m.OPEX_SOEL) + m.OPEX_Aircooler) <= 1e-6)



# ********************* OBJECTIVE FUNCTION ********************************** #
m.TAC = Objective(expr = (((m.CAPEX_FAN + m.trf_out["CAPEX_DFM"] + 
                                         m.CAPEX_SOEL + m.CAPEX_Boiler
                                         + m.CAPEX_Aircooler + m.CAPEX_Separator + m.CAPEX_PVessel) * ACCR) + (m.OPEX_FAN + (m.trf_out["OPEX_DFM"]) 
                                                                + (m.OPEX_SOEL) + m.OPEX_Aircooler)) - m.Profit, sense = minimize)

# for v in m.component_objects(Var, active=True):
#     for idx in v:
#         print(f"{v.name}[{idx}] = {v[idx].value}")
        
# #------------------- Linearization / Surrogate -------------------
# x0 = [1000.0, 434.136, 104.0, 26.0, 26.0, 4.0, 1.76955, 0]  # example point
# f0 = TRF_blackbox(x0)
# grad0 = grad_TRF_blackbox(x0)

# use_linear_basis = True

# def basis_rule(component, ef_expr):
#     """Linear approximation of an ExternalFunction around x0"""
#     if not use_linear_basis:
#         return 0
#     pycb = getattr(ef_expr, "_fcn", None)
#     if pycb is None:
#         return 0
#     idx = m._ef_pycb_to_idx.get(pycb, None)
#     if idx is None:
#         return 0
#     # linearized surrogate: f0 + grad*(args - x0)
#     expr = f0[idx]
#     for i, xi in enumerate(TRF_inputs):
#         expr += grad0[idx][i] * (ef_expr.arg(i) - x0[i])
#     return expr

# # --- Zero basis surrogate --------------------------------------------------
basis_rule = lambda comp, ef: 0


def DEGREES_OF_FREEDOM(model):
    """Decision variables on which the trust region is imposed (thesis Section 8.2)."""
    return [model.t_ads, model.T[5], model.x[5, 'H2'], model.Q_air, model.Q_DFM,
            model.D, model.L]
