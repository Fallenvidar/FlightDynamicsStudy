# imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Variables

# mass related

# gravity
g = 9.81
# mass vehicle(kg)
m_v = 200
# mass fuel (kg)
m_f = 800
# mass flow rate (kg/s)
mm = 5.1
# Exhaust velocity m/s
ve = 2943
# drag related
# surface density
rho = 1.225

# Scale height
H = 8500

# Area(m)
A = 1.0

# Areqa(m) wing
Aw =

# Angle of Ascend
theta =

# Coefficient of Drag
Cd = 0.3

# Coefficient of Lifit
CL =

# Initial conditions
# case 1
y0 = 0
v0 = 0
F0 = mm * ve
m_f0 = m_f
state0 = [y0, v0, m_f0]
t_span = (0, 1000)

#Non-Thrust Vectoring Case
def case2a(t, state):
    y,x,vy,vx,m_f = state

    # Controls mass and fuel
    if m_f > 0:
        Ft = F0
        mt = m_v + m_f
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    dydt = vy
    dxdt = vx

    dvydt = (Ft * np.sin(theta) - mt * g - ((1 / 2) * rho * np.exp(-y / H) * v * abs(v) * A * Cd) + (1/2 * rho * v * abs(v) * Aw ) / mt

    dxydt =

    return[dydt, dxdt ,dvydt, dxydt,dm_fuel_dt]

#Thrust Vectoring Case
def case2b(t, state):
    y,x,vy,vx, m_fuel = state

    if m_f > 0:
        Ft = F0
        mt = m_v + m_f
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    dydt = vy
    dxdt = vx

    dvydt = (Ft * np.sin(theta) - mt * g - ((1 / 2) * rho * np.exp(-y / H) * v * abs(v) * A * Cd) + (1/2 * rho * v * abs(v) * Aw ) ) / mt

    dxydt =

    return [dydt, dxdt, dvydt, dxydt, dm_fuel_dt]
