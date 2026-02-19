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

# Coefficient of Drag
C = 0.3

# Initial conditions
# case 1
y0 = 0
v0 = 0
F0 = mm * ve
m_f0 = m_f
state0 = [y0, v0, m_f0]
t_span = (0, 1000)


# Case 1 vertical particle at a certain thrust
def Case1(t, state):
    y, v, m_f = state

    # Controls mass and fuel
    if m_fuel > 0:
        Ft = F0
        mt = m_v + m_fuel
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    # ODE
    dydt = v
    dvdt = (Ft - mt * g - ((1 / 2) * rho * np.exp(-y / H) * v * abs(v) * A * C)) / mt

    return [dydt, dvdt, dm_fuel_dt]

#Non-Thrust Vectoring Case
def case2a(t, state):
    y,x,v, m_f = state

    # Controls mass and fuel
    if m_f > 0:
        Ft = F0
        mt = m_v + m_f
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    return[dydt, dxdt ,dvdt, dm_fuel_dt]

#Thrust Vectoring Case
def case2b(t, state):
    y,x,v, m_fuel = state

    if m_f > 0:
        Ft = F0
        mt = m_v + m_f
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    return [dydt, dxdt, dvdt, dm_fuel_dt]



# Check for when the ground is hit after fuel is totally consumed
def hit_ground(t, state):
    y, v, m_fuel = state
    return y  # event occurs when y = 0


hit_ground.terminal = True  # stop integration
hit_ground.direction = -1  # only trigger when y is decreasing


# Check for max height (apogee) - when velocity = 0
def max_height(t, state):
    y, v, m_fuel = state
    return v  # event occurs when v = 0


max_height.terminal = False  # don't stop integration
max_height.direction = -1  # only trigger when v is decreasing (going from + to 0)


# Check for max velocity - when acceleration = 0
def max_velocity(t, state):
    y, v, m_fuel = state

    # Recalculate acceleration to find when it crosses zero
    if m_fuel > 0:
        Ft = F0
        mt = m_v + m_fuel
    else:
        Ft = 0
        mt = m_v

    a = (Ft - mt * g - ((1 / 2) * rho * np.exp(-y / H) * v * abs(v) * A * C)) / mt

    return a  # event occurs when a = 0


max_velocity.terminal = False  # don't stop integration
max_velocity.direction = -1  # only trigger when a goes from + to 0

# solver
solution = solve_ivp(
    Case1,
    t_span,
    state0,
    events=[hit_ground, max_height, max_velocity],
    max_step=1
)

#output

t = solution.t
y = solution.y[0]
v = solution.y[1]

#give information on max velocity, height, and impact time
if solution.t_events[2].size > 0:
    print("Max velocity time (s):", round(solution.t_events[2][0]))
    print("Max velocity (m/s):", round(solution.y_events[2][0][1]))

if solution.t_events[1].size > 0:
    print("Max height time (s):", round(solution.t_events[1][0]))
    print("Max height (m):", round(solution.y_events[1][0][0]))

if solution.t_events[0].size > 0:
    print("Impact time (s):", round(solution.t_events[0][0]))
    print("Impact velocity (m/s):", round(abs(solution.y_events[0][0][1])))

# identify when max velocity is achived, max height, and imapct
if solution.t_events[2].size > 0:  # Max velocity
    t_max_v = solution.t_events[2][0]
    y_max_v = solution.y_events[2][0][0]
    plt.plot(t_max_v, y_max_v, 'ro', markersize=4, label='Max Velocity')

if solution.t_events[1].size > 0:  # Max height
    t_max_h = solution.t_events[1][0]
    y_max_h = solution.y_events[1][0][0]
    plt.plot(t_max_h, y_max_h, 'go', markersize=4, label='Max Height')

if solution.t_events[0].size > 0:  # Impact
    t_impact = solution.t_events[0][0]
    y_impact = solution.y_events[0][0][0]
    plt.plot(t_impact, y_impact, 'ko', markersize=4, label='Impact')

plt.legend()
plt.plot(t, y)
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.grid()
plt.show()