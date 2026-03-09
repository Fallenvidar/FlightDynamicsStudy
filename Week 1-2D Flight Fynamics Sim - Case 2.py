# imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# Gen Variables
g = 9.81
rho = 1.225
H = 8500

# F-16 Inspired parameters( Double check )(not after burner at military thrust)
m_v = 8500      # kg dry mass (actual F-16 is ~8,500 kg)
m_f = 3300      # kg fuel (internal fuel load)
mm = 25.3       # kg/s fuel flow at military thrust
ve = 3000       # m/s effective exhaust velocity
A = 27.87       # m² wing reference area
Aw = 27.87      # same reference area for lift
CL = 0.5        # cruise lift coefficient
Cd = 0.02       # clean configuration drag coefficient

# Initial conditions
# case 1
y0 = 0
x0 =0
# helps to prevent sinking caused by gravity at start. Sacrifice since this is a particle ,and we are focusing on
# the relationship between thrust vectoring and non-thrust vectoring
vy0= 1.0
vx0 = 0
F0 = mm * ve
m_f0 = m_f
state0 = [y0, x0,vy0,vx0, m_f0]
t_span = (0, 5000)
theta_b = np.radians(45)
theta_v = 0
target = 5000
max_angle = np.radians(60)

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

    v_mag = np.sqrt(vx**2 + vy**2)
    # Theta Body change
    #Sacrfie for min launch angle to align with the fact this is a simulation it not suppose to be accuracy just get what we need out of it
    min_angle = np.radians(45)  # minimum launch angle
    if y < target / 2:
        # merge progress into base equation
        progress = y / (target / 2)
        theta_b = min_angle + (max_angle - min_angle) * progress
    elif y < target:
        progress = (y - target / 2) / (target / 2)
        theta_b = max_angle * (1 - progress)
    elif m_f > 0: # cruise, still have fuel
        theta_b = 0
    else:
        theta_b = -np.radians(15) # descent, fuel gone

    theta_v = np.arctan2(vy, vx)
    #Outputs
    dydt = vy
    dxdt = vx
    dvydt = (Ft * np.sin(theta_b)- mt * g - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * A * Cd) * np.sin(theta_v) + (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * Aw * CL) * np.cos(theta_v)) / mt
    dvxdt = (Ft * np.cos(theta_b) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * A * Cd) * np.cos(theta_v) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * Aw * CL) * np.sin(theta_v)) / mt


    return[dydt, dxdt ,dvydt, dvxdt,dm_fuel_dt]

#Thrust Vectoring Case
def case2b(t, state):
    y,x,vy,vx, m_fuel = state

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

    dvydt = (Ft * np.sin(theta) - mt * g - ((1 / 2) * rho * np.exp(-y / H) * v * abs(v) * A * Cd) + (1/2 * rho * v * abs(v) * Aw ) ) / mt
    dvxdt = (Ft * np.cos(theta_b) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * A * Cd) * np.cos(theta_v) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * Aw * CL) * np.sin(theta_v)) / mt

    return [dydt, dxdt, dvydt, dxydt, dm_fuel_dt]




#*Events*
# Check for when the ground is hit after fuel is totally consumed
def hit_ground(t, state):
    y, x, vy, vx, m_f = state
    return y - 0.1  # event occurs when y = 0


hit_ground.terminal = True  # stop integration
hit_ground.direction = -1  # only trigger when y is decreasing


solution = solve_ivp(
    case2a,
    t_span,
    state0,
    events=[hit_ground],
    max_step=1
)

# Extract states
t = solution.t
y = solution.y[0]
x = solution.y[1]
vy = solution.y[2]
vx = solution.y[3]




# Graphing
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Altitude')
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Trajectory')
plt.xlabel("Downrange Distance (m)")
plt.ylabel("Altitude (m)")
plt.grid()
plt.legend()
plt.show()