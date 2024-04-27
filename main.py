# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import numpy as np
from scipy.optimize import minimize
from scipy.constants import g
from tabulate import tabulate

RE_KM = 6371                 # radius earth
MU = 3.986005 * 10**5       # Standard gravitational parameter of Earth in km^3/s^2


def semi_major_axis_km(apogee_km, perigee_km):
    """
    returns the semi major axis (in km) from the apogee and perigee input (both in km)
    """
    return(apogee_km + RE_KM + perigee_km + RE_KM) / 2


def orbital_speed_meter(distance_km, sma_km):
    """
    returns the orbital speed in m/s of an object at distance kilometers
    in an orbit with semi-major axis sma in kilometers
    """
    return 1000 * math.sqrt(MU * ((2.0/(distance_km + RE_KM)) - (1.0/sma_km)))


def delta_v(v1, v2, inc_diff):
    """
    returns the delta velocity in m/s of an object traveling at velocity v1 in
    a particular point in it's orbit (e.g. apogee), to v2 with an inclination
    difference of inc_diff in degrees
    """
    return math.sqrt(v1**2 + v2**2 - 2 * v1 * v2 * math.cos(math.radians(inc_diff)))


def rocket(dv, isp_eff):
    """
    Use the rocket equation to solve for the multiplier
    """
    return math.exp(dv / (isp_eff * g))


def objective_function(dry_mass_input, maneuvers):
    """
    Calculate a Propellant Budget given the dry mass of a satellite and a list of maneuvers.
    :param dry_mass_input: Dry mass of the satellite (typically in kg)
    :param maneuvers:
    1. "Name of Maneuver",
    2. 'add'; 'dvmono'; dvprop';
        if add, paramenter 3 is fuel mass to add, parameter4 is ox mass, parameter5 = 'na'
        if dvmono parameter3 = delta V; paramenter4 = Effective Isp parameter5 = 'na'
        if dvbiprop parameter3 = delta V; parameter4 = Effective Isp parameter5 = mixture ratio
    :return: fuel, ox and mass for each manuever

    add fuel/ox: Residuals (mass left at end of mission)
    dVmono: Manuevers that are executed by REAs (mono prop. - e.g. hydrazine). The mass calculated by the
         rocket equation is all fuel in this case.
    dVbiprop: executed by bi-prop engines (fuel and ox). Rocket equation mass is split between fuel and
         oxidizer based on the mixture ratio.

    Assuming a Payload Adapter Mass of 95 kg

    """
    list_size = len(maneuvers) + 1
    fuel_budget = np.zeros(list_size)
    ox_budget = np.zeros(list_size)
    mass_budget = np.zeros(list_size)

    mass_budget[0] = dry_mass_input

    for index, sublist in enumerate(maneuver_list):
        index_plus_1 = index + 1
        if sublist[1] == 'add':
            fuel_budget[index_plus_1] = sublist[2]
            ox_budget[index_plus_1] = sublist[3]
            mass_budget[index_plus_1] = mass_budget[index] + fuel_budget[index_plus_1] + ox_budget[index_plus_1]
        elif sublist[1] == 'dVmono':
            fuel_budget[index_plus_1] = mass_budget[index] * (rocket(sublist[2], sublist[3]) - 1)
            mass_budget[index_plus_1] = mass_budget[index] + fuel_budget[index_plus_1]
        elif sublist[1] == 'dVbiprop':
            biprop_mass = mass_budget[index] * (rocket(sublist[2], sublist[3]) - 1)
            fuel_budget[index_plus_1] = biprop_mass / (1 + sublist[4])
            ox_budget[index_plus_1] = fuel_budget[index_plus_1] * sublist[4]
            # print(f'sublist = {sublist}, biprop_mass, fuel, ox {biprop_mass} {fuel[index_plus_1]} {ox[index_plus_1]}')
            mass_budget[index_plus_1] = mass_budget[index] + fuel_budget[index_plus_1] + ox_budget[index_plus_1]
        else:
            print("Unknown value in second position of sublist:", sublist[1])

    payload_launch_adapter = 95
    total_fuel = sum(fuel_budget)
    total_ox = sum(ox_budget)
    separated_mass = dry_mass_input + pressurant + total_fuel + total_ox
    total_mass = separated_mass + payload_launch_adapter
    #    print(f'{total_mass} = {dry_mass} {pressurant} {total_fuel} {total_ox} {separated_mass}')
    return total_mass, total_fuel, total_ox, fuel_budget, ox_budget, mass_budget


def real_objective():
    [total_mass, _totalFuel, _totalOx, _fuel, _ox, _mass] = objective_function(dry_mass_input, maneuver_list)
    return -total_mass


# Constraint functions fuel
def constraint_fuel():
    [_totalMass, total_fuel, _totalOx, _fuel, _ox, _mass] = objective_function(dry_mass_input, maneuver_list)
    return 2100 - total_fuel


def constraint_ox():
    [_a, _b, total_ox, _c, _d] = objective_function(dry_mass_input, maneuver_list)
    return 1100 - total_ox


def constraint_wetmass():
    [total_mass, _totalFuel, _totalOx, _fuel, _ox, _mass] = objective_function(dry_mass_input, maneuver_list)
    return 6230 - total_mass


def print_prop_budget(fuel_array, ox_array, mass_array):
    table_data = [["Fuel", "Ox", "Mass"]]
    for fuel_maneuver, ox_maneuver, mass_maneuver in zip(fuel_array, ox_array, mass_array):
        table_data.append([fuel_maneuver, ox_maneuver, mass_maneuver])
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    apogee_init = 35786
    perigee_init = 185
    inc_init = 27
    sma_init = semi_major_axis_km(apogee_init, perigee_init)
    vel_1 = orbital_speed_meter(apogee_init, sma_init)
    print(f'vel_1 = {vel_1:0.1f} sma = {sma_init:0.1f}')

    apogee_fin = 35786
    perigee_fin = 35786
    inc_fin = 0
    sma_fin = semi_major_axis_km(apogee_fin, perigee_fin)
    vel_2 = orbital_speed_meter(apogee_fin, sma_fin)
    print(f'vel_2 = {vel_2:0.1f} sma = {sma_fin:0.1f}')

    del_vel = delta_v(vel_1, vel_2, inc_init - inc_fin)
    print(f'del_vel = {del_vel:0.5f}')

    dry_mass = 3324
    pressurant = 1

    delV_xfer = 45
    delV_ewsk = 36
    delV_repositions = 90
    delV_margin = 135
    delV_disposal = 10

    # Inputs based on Isp and location of thrusters/engines
    Isp_eff_5lbfrea = 214
    Isp_eff_pt2lbfrea = 207

    # Inputs from somewhere
    residuals_fuel = 5
    mom_unload = 35
    separation_attitude = 5

    # Minimize the aggregated objective function with constraints
    maneuver_list = [['Residuals', 'add', residuals_fuel, 0, 'na'],
                     ['TO Contingency', 'add', 4, 0, 'na'],
                     ['Momentum', 'add', mom_unload, 0, 'na'],
                     ['Disposal', 'dVmono', delV_disposal, Isp_eff_5lbfrea, 'na'],
                     ['Margin', 'dVmono', delV_margin, Isp_eff_pt2lbfrea, 'na'],
                     ['Repositions', 'dVmono', delV_repositions, Isp_eff_5lbfrea, 'na'],
                     ['EWSK', 'dVmono', delV_ewsk, Isp_eff_pt2lbfrea, 'na'],
                     ['Transfer Orbit', 'dVmono', delV_xfer, Isp_eff_5lbfrea, 'na'],
                     ['Separation and Attitude Slews', 'add', separation_attitude, 0, 'na'],
                     ]
    [wet_mass, tot_fuel, _tot_ox, fuel, ox, mass] = objective_function(dry_mass + pressurant, maneuver_list)
    print(f'Wet Mass: {wet_mass} Fuel: {tot_fuel} Separated Mass:{mass[-1]}')
    print_prop_budget(fuel, ox, mass)

# -------------------------------

delV_settling = 1.43
delV_xfer = 1214.47
delV_att_control = 12.27
delV_xfer_contingency = 8
delV_repo_MOL = 22.78
delV_nssk = 710.4
delV_ewsk = 49
delV_ewsk_bu = 1.6
delV_rapid_repo = 80
delV_disposal = 11

# Inputs based on Isp and location of thrusters/engines
Isp_eff_aj_nssk = 505.7  # aj location optimized for nssk
Isp_eff_aj_ewsk = 451.2  # aj location optimized for nssk
Isp_eff_ewsk_bu = 124.4
Isp_eff_leros_oi = 321.8  # Leros performance with high eff engine placement
Isp_eff_rea = 218.8  # 5lbf performance with high eff thruster placement
Isp_eff_rea_att_con = 206.8  # efficiency with pulsing for attitude control
mixture_ratio = 0.85

# Inputs from somewhere
pressurant = 11.5
residuals_fuel = 33.49
residuals_ox = 9.4
unaug_penalty = 9.55  # mass penalty running arcjets in non-power mode (unaugmented)
mom_unload = 1.5  # must be from GN&C seems constants
orbit_test = 20.76
separation_attitude = 3

# Minimize the aggregated objective function with constraints
maneuver_list = [['Residuals', 'add', residuals_fuel, residuals_ox, 'na'],
                 ['Disposal', 'dVmono', delV_disposal, Isp_eff_aj_ewsk, 'na'],
                 ['Rapid Relocation', 'dVmono', delV_rapid_repo, Isp_eff_rea, 'na'],
                 ['S/K unaugmented penalties', 'add', unaug_penalty, 0, 'na'],
                 ['Momentum', 'add', mom_unload, 0, 'na'],
                 ['EW Backup', 'dVmono', delV_ewsk_bu, Isp_eff_ewsk_bu, 'na'],
                 ['EWSK', 'dVmono', delV_ewsk, Isp_eff_aj_ewsk, 'na'],
                 ['NSSK', 'dVmono', delV_nssk, Isp_eff_aj_nssk, 'na'],
                 ['Repositioning', 'dVmono', delV_repo_MOL, Isp_eff_aj_ewsk, 'na'],
                 ['In-orbit Test', 'add', orbit_test, 0, 'na'],
                 ['Transfer Orbit contingency', 'dVbiprop', delV_xfer_contingency, Isp_eff_leros_oi, mixture_ratio],
                 ['Attitude Control During LAE', 'dVmono', delV_att_control, Isp_eff_rea_att_con, 'na'],
                 ['Transfer Orbit', 'dVbiprop', delV_xfer, Isp_eff_leros_oi, mixture_ratio],
                 ['Settling Burns', 'dVmono', delV_settling, Isp_eff_rea, 'na'],
                 ['Separation and Attitude Slews', 'add', separation_attitude, 0, 'na'],
                 ]

# Define the constraints as a list of dictionaries
# Constraints
MAX_FUEL = 2100
MAX_OX = 1200
MAX_LV_CAPABILITY_GTO_1500 = 6230
constraints = [{'type': 'ineq', 'fun': constraint_fuel},
               {'type': 'ineq', 'fun': constraint_ox},
               {'type': 'ineq', 'fun': constraint_wetmass}
               ]
#
initial_guess = np.array([2000])
# result = minimize(real_objective, initial_guess, constraints=constraints)
dry_mass_input = 1000
# result = minimize(real_objective, 2000)

# --------------- Print Results -----------------

[wet_mass, tot_fuel, tot_ox, fuel, ox, mass] = objective_function(3300 + pressurant, maneuver_list)
print(f'Wet Mass with PLA: {wet_mass} Separated Mass:{mass[-1]} Fuel: {tot_fuel} ox: {tot_ox}')
print_prop_budget(fuel, ox, mass)
