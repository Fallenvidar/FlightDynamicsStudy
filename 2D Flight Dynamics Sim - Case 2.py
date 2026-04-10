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
mm = 10      # kg/s fuel flow at military thrust
ve = 3000       # m/s effective exhaust velocity
A = 27.87       # m² wing reference area
Aw = 27.87      # same reference area for lift
CL = 1.5      # cruise lift coefficient
Cd = 0.02       # clean configuration drag coefficient

# Initial conditions
# case 1
# starts in the air so lift will counteract gravity property
y0 = 100
x0 =0
# helps to prevent sinking caused by gravity at start. Sacrifice since this is a particle ,and we are focusing on
# the relationship between thrust vectoring and non-thrust vectoring
# have at min velocity to produce thrust modeling the takeoff point from a runway
vy0= 10
vx0 = 120
F0 = mm * ve
m_f0 = m_f
state0 = [y0, x0,vy0,vx0, m_f0]
t_span = (0, 1500)
theta_b = np.radians(45)
theta_v = 0
target = 2500
max_angle = np.radians(60)
#identifying phase
phase = [1]
# to record what happening at different phases
phase_log = []

min_angle = np.radians(45)

#Non-Thrust Vectoring Case
def case2a(t, state):
    y, x, vy, vx, m_f = state

    # Controls mass and fuel
    if m_f > 0:
        Ft = F0
        mt = m_v + m_f
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    v_mag = np.hypot(vx, vy)

    #---------------------------------------- Clean up comments before finishing
    #Phase Logic
    # Theta Body change and angle stuff(put in a list for theta b so can do a theta body vs time graph)
    #Sacrfie for min launch angle to align with the fact this is a simulation it not suppose to be accuracy just get what we need out of it

    #Restoing phase idea that there will be no oscillation
    #Using a list instead so there are no statefulness problems
    min_angle = np.radians(45)  # minimum launch angle
    if phase[0] == 1 and y >= target / 2:
        phase[0] = 2
        phase_log.append((2, t, y, v_mag))
        print(f"Phase 2 entered at y={y:.1f}m, v_mag={v_mag:.1f}m/s, t={t:.1f}s")
    if phase[0] == 2 and y >= target:
        phase[0] = 3
        phase_log.append((3, t, y, v_mag))
        print(f"Phase 3 entered at y={y:.1f}m, v_mag={v_mag:.1f}m/s, t={t:.1f}s")
    if phase[0] == 3 and m_f <= mm:
        phase[0] = 4
        phase_log.append((4, t, y, v_mag))
        print(f"Phase 4 entered at y={y:.1f}m, v_mag={v_mag:.1f}m/s, t={t:.1f}s")


    if phase[0] == 1:
        theta_b = min_angle + (max_angle - min_angle) * (y / (target / 2))
    elif phase[0] == 2:
        theta_b = max_angle * (1 - ((y - target / 2) / (target / 2)))
        theta_b = max(theta_b, 0)  # never let it go negative in phase 2
    elif phase[0] == 3:
        theta_b = np.radians(5) #Chandged
    else:
        theta_b = -np.radians(15) # descent, fuel gone


    #May need to remove serves as a way to make sure theta_v doesn't change crazy at low v
    if v_mag < 20.0:  # below meaningful aerodynamic speed
        theta_v = theta_b  # align with body, forces act along thrust direction
    elif vx <= 0: #ADDED
        theta_v = theta_b
    else:
        theta_v = np.arctan2(vy, vx)

    # -------------- Removed/Replaced likley cause of the looping
    # Angle of attack is difference between body and velocity direction
    alpha = theta_b - theta_v

    #To deal with drag being greater than thrust as small v
    # CL varies linearly with angle of attack up to stall
    # At alpha = 0 (cruise): CL = CL_min
    # At alpha = max (steep climb): CL = CL_max
    # CL_min = 0.3
    # CL_max = 1.5
    # alpha_max = np.radians(20)  # stall angle

    # Linear variation, clamped at stall
    # alpha_clamped = np.clip(abs(alpha), 0, alpha_max)
    # CL_eff = CL_min + (CL_max - CL_min) * (alpha_clamped / alpha_max)
    # -----------------

    #Added
    # ----------------------------------------
    # FIX 1 — speed dependent CL cap via structural load factor limit
    # Replaces bare CL constant in lift calculation
    # Caps lift to 9g structural limit, preventing unrealistic forces at high speed
    rho_a = rho * np.exp(-y / H)
    q = max(0.5 * rho_a * v_mag**2, 1.0)  # minimum 1 Pa dynamic pressure

    L_max = 9.0 * mt * g
    if q * Aw > 0:
        CL_max_speed = L_max / (q * Aw)
    else:
        CL_max_speed = CL

    # ----------------------------------------
    # FIX 2 — angle of attack dependent CL
    # Replaces constant CL throughout all phases
    # CL varies continuously with alpha rather than switching by phase
    alpha = abs(theta_b - theta_v)
    alpha_max = np.radians(20)
    CL_min = 0.3
    CL_max = 1.5
    CL_alpha = CL_min + (CL_max - CL_min) * min(alpha / alpha_max, 1.0)

    # Combined CL — takes the more restrictive of aerodynamic and structural limits
    # This is where Fix 1 and Fix 2 meet
    CL_eff = min(CL_alpha, CL_max_speed)

    #----------------------------------------
    # Combing things together before final equation lets go with this method for now on to help simplifiy
    # Force equations — use CL_eff replacing bare CL
    # D and L now computed as named variables for clarity
    D = q * A * Cd
    L = q * Aw * CL_eff

    #Outputs
    dydt = vy
    dxdt = vx
    dvydt = (Ft * np.sin(theta_b) - mt * g - D * np.sin(theta_v) + L * np.cos(theta_v)) / mt
    dvxdt = (Ft * np.cos(theta_b) - D * np.cos(theta_v) - L * np.sin(theta_v)) / mt

    # old
    # dvydt = (Ft * np.sin(theta_b)- mt * g - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * A * Cd) * np.sin(theta_v) + (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * Aw * CL_eff) * np.cos(theta_v)) / mt
    # dvxdt = (Ft * np.cos(theta_b) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * A * Cd) * np.cos(theta_v) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * Aw * CL_eff) * np.sin(theta_v)) / mt


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
    dvxdt = (Ft * np.cos(theta_b) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * A * Cd) * np.cos(theta_v) - (1/2 * rho * np.exp(-y/H) * v_mag * abs(v_mag) * Aw * CL_eff) * np.sin(theta_v)) / mt

    return [dydt, dxdt, dvydt, dxydt, dm_fuel_dt]




#*Events*
# Check for when the ground is hit after fuel is totally consumed
def hit_ground(t, state):
    y, x, vy, vx, m_f = state
    return y - 0.1  # event occurs when y = 0


hit_ground.terminal = True  # stop integration
hit_ground.direction = -1  # only trigger when y is decreasing

phase[0] = 1
solution = solve_ivp(
    case2a,
    t_span,
    state0,
    events=[hit_ground],
    max_step=1.0,
    method='LSODA'
)

# Extract states
t = solution.t
y = solution.y[0]
x = solution.y[1]
vy = solution.y[2]
vx = solution.y[3]

#solving v-mag over time
v_mag_sol = np.hypot(vx, vy)

# Extract phase transition points for markers
phase_times = [entry[1] for entry in phase_log]
phase_alts  = [entry[2] for entry in phase_log]
phase_nums  = [entry[0] for entry in phase_log]
phase_vmags = [entry[3] for entry in phase_log]

# Graphing
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Altitude', color='blue')
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, py),
                xytext=(pt + 20, py + 200),
                fontsize=8,
                color='red')
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

plt.figure(figsize=(10, 6))
plt.plot(t, vx, label='vx (horizontal)', color='green')
plt.plot(t, vy, label='vy (vertical)', color='orange')
for pt in phase_times:
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Components vs Time — Case 2A")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, v_mag_sol, label='Total Speed', color='purple')
plt.axhline(y=75, color='red', linestyle='--',
            label='Min cruise speed (~75 m/s)', alpha=0.7)
for pt in phase_times:
    plt.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Total Speed vs Time — Case 2A")
plt.grid()
plt.legend()
plt.show()

#checking theta_b and theta_v
# Recompute theta_v from solution
theta_v_history = np.arctan2(vy, vx)

# Recompute theta_b from solution using same phase logic
theta_b_history = np.zeros(len(t))
phase_recon = 1  # reconstructed phase

for i in range(len(t)):
    y_i = y[i]
    m_f_i = solution.y[4][i]

    # Phase reconstruction - forward only
    if phase_recon == 1 and y_i >= target / 2:
        phase_recon = 2
    if phase_recon == 2 and y_i >= target:
        phase_recon = 3
    if phase_recon == 3 and m_f_i <= mm:
        phase_recon = 4

    # Theta_b from phase
    if phase_recon == 1:
        progress = y_i / (target / 2)
        theta_b_history[i] = min_angle + (max_angle - min_angle) * progress
    elif phase_recon == 2:
        progress = (y_i - target / 2) / (target / 2)
        theta_b_history[i] = max(max_angle * (1 - progress), 0)
    elif phase_recon == 3:
        theta_b_history[i] = np.radians(5)
    else:
        theta_b_history[i] = np.radians(-15)

# Convert to degrees for readability
theta_b_deg = np.degrees(theta_b_history)
theta_v_deg = np.degrees(theta_v_history)

# Plot theta_b and theta_v over time
plt.figure(figsize=(10, 6))
plt.plot(t, theta_b_deg, label='theta_b (body angle)', color='blue')
plt.plot(t, theta_v_deg, label='theta_v (velocity angle)', color='orange')
for pt in phase_times:
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Theta Body vs Theta Velocity — Case 2A")
plt.grid()
plt.legend()
plt.show()