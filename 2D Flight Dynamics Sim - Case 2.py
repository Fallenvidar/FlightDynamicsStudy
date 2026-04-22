# imports
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create timestamped output folder for this run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("simulation_outputs", f"run_{run_timestamp}")
os.makedirs(output_dir, exist_ok=True)
print(f"Saving plots to: {output_dir}")

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
CL = 0.5      # cruise lift coefficient
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
state0 = [y0, x0,vy0,vx0, m_f0, np.radians(45)]
t_span = (0, 2500)
theta_b = np.radians(45)
theta_v = 0
target = 2500
max_angle = np.radians(60)
#identifying phase
phase = [1]
# to record what happening at different phases
phase_log = []

min_angle = np.radians(45)

def save_and_show(filename, dpi=150):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()
    plt.close()  # prevents memory buildup across many plots

#Non-Thrust Vectoring Case
def case2a(t, state):
    y, x, vy, vx, m_fuel, theta_b = state

    # ---------- Mass & Thrust ----------
    if m_fuel > 0:
        Ft = F0
        mt = m_v + m_fuel
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    v_mag = np.hypot(vx, vy)

    # ---------- Atmosphere ----------
    rho_a = rho * np.exp(-y / H)
    q = max(0.5 * rho_a * v_mag ** 2, 1.0)

    # ---------- Phase Transitions (with logging) ----------
    if phase[0] == 1 and y >= target / 2:
        phase[0] = 2
        if 2 > (phase_log[-1][0] if phase_log else 0):
            phase_log.append((2, t, y, v_mag))
            print(f"Phase 2 at y={y:.1f}m, v={v_mag:.1f}m/s, t={t:.1f}s")

    if phase[0] == 2 and y >= target:
        phase[0] = 3
        if 3 > (phase_log[-1][0] if phase_log else 0):
            phase_log.append((3, t, y, v_mag))
            print(f"Phase 3 at y={y:.1f}m, v={v_mag:.1f}m/s, t={t:.1f}s")

    if phase[0] == 3 and m_fuel <= mm:
        phase[0] = 4
        if 4 > (phase_log[-1][0] if phase_log else 0):
            phase_log.append((4, t, y, v_mag))
            print(f"Phase 4 at y={y:.1f}m, v={v_mag:.1f}m/s, t={t:.1f}s")

    # ---------- Flight Path Angle ----------
    if v_mag < 20.0 or vx <= 0:
        theta_v = theta_b
    else:
        theta_v = np.arctan2(vy, vx)

    # ---------- Aerodynamic Coefficients (α‑dependent lift, speed‑dependent drag) ----------
    alpha = theta_b - theta_v
    alpha_deg = np.degrees(alpha)

    Cd_base = 0.02
    if v_mag > 150:
        Cd_speed = Cd_base + 0.005 * (v_mag - 150) / 50
        Cd_speed = min(Cd_speed, 0.035)
    else:
        Cd_speed = Cd_base

    CL_alpha = 0.08
    CL_max = 1.2
    CL = CL_max * np.tanh(CL_alpha * alpha_deg / CL_max)
    CL = 0.7 * 0.5 + 0.3 * CL
    k_ind = 0.1
    Cd = Cd_speed + k_ind * CL**2

    D = q * A * Cd
    L = q * Aw * CL

    # ---------- Theta_b Target Calculation ----------
    if phase[0] == 1:
        theta_b_target = min_angle + (max_angle - min_angle) * (y / (target / 2))
    elif phase[0] == 2:
        # Scheduled theta_b reduction (same as before)
        theta_b_scheduled = max_angle * (1 - (y - target / 2) / (target / 2))
        theta_b_scheduled = theta_b_scheduled - np.radians(2) * (y - target / 2) / (target / 2)

        # Desired vertical acceleration to reduce vy to zero
        # Gain increases as we approach target altitude
        progress = (y - target / 2) / (target / 2)  # 0 → 1
        K_vy = 0.2 * progress  # damping gain grows
        ay_desired = -K_vy * vy  # oppose current vertical speed

        # Required vertical force
        F_aero_y = L * np.cos(theta_v) - D * np.sin(theta_v)
        Fy_needed = mt * (g + ay_desired) - F_aero_y

        if Ft > 0:
            # Blend scheduled angle with trim needed for ay_desired
            theta_b_trim = np.arctan(Fy_needed / Ft)
            theta_b_trim = np.clip(theta_b_trim, np.radians(-10), np.radians(60))
            # Use a mix: mostly scheduled, but with correction
            blend = progress ** 2  # more correction near target
            theta_b_target = (1 - blend) * theta_b_scheduled + blend * theta_b_trim
        else:
            theta_b_target = theta_b_scheduled
    elif phase[0] == 3:
        # Trim calculation (same as before)
        Ft = F0 * 0.7
        alt_error = y - target
        ay_target = -0.005 * alt_error - 0.05 * vy
        F_aero_y = L * np.cos(theta_v) - D * np.sin(theta_v)
        Fy_needed = mt * (g + ay_target) - F_aero_y
        if Ft > 0:
            theta_b_target = np.arctan(Fy_needed / Ft)
            theta_b_target = np.clip(theta_b_target, np.radians(-3), np.radians(5))
            theta_b_target = np.clip(theta_b_target, np.radians(-3), np.radians(5))
        else:
            Ft = F0
            theta_b_target = -np.arctan(15)
    else:   # Phase 4
        # For phugoid demonstration: hold theta_b constant (no target change)
        theta_b_target = theta_b

    # ---------- Rate Limit Only (No Filter) ----------
    if phase[0] != 4:
        max_rate = np.radians(10)   # 10 deg/s max rate
        dtheta_b_dt = np.clip(theta_b_target - theta_b, -max_rate, max_rate)
    else:
        dtheta_b_dt = 0.0          # freeze theta_b during glide

    # ---------- Equations of Motion ----------
    dydt = vy
    dxdt = vx
    dvydt = (Ft * np.sin(theta_b) - mt * g - D * np.sin(theta_v) + L * np.cos(theta_v)) / mt
    dvxdt = (Ft * np.cos(theta_b) - D * np.cos(theta_v) - L * np.sin(theta_v)) / mt

    return [dydt, dxdt, dvydt, dvxdt, dm_fuel_dt, dtheta_b_dt]

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
    y, x, vy, vx, m_f,theta_b  = state
    return y - 0.1
    # event occurs when y = 0


hit_ground.terminal = True  # stop integration
hit_ground.direction = -1  # only trigger when y is decreasing

phase[0] = 1
phase_log = []
solution = solve_ivp(
    case2a,
    t_span,
    state0,
    events=[hit_ground],
    max_step=1.0,
    method='RK45'
)

# Diagnostics
print(f"Termination status: {solution.status}")
print(f"Termination message: {solution.message}")
print(f"Final time: {solution.t[-1]:.2f}s")
print(f"Final altitude: {solution.y[0][-1]:.2f}m")
print(f"Final vx: {solution.y[3][-1]:.2f}m/s")

# Extract states
t = solution.t
y = solution.y[0]
x = solution.y[1]
vy = solution.y[2]
vx = solution.y[3]
m_fuel_sol = solution.y[4]
theta_b_sol = solution.y[5]  # directly from state — no reconstruction needed

# Solving v_mag over time
v_mag_sol = np.hypot(vx, vy)

# Extract phase transition points for markers
phase_times = [entry[1] for entry in phase_log]
phase_alts  = [entry[2] for entry in phase_log]
phase_nums  = [entry[0] for entry in phase_log]
phase_vmags = [entry[3] for entry in phase_log]

# Theta_v recomputed from solution velocity components
# Theta_b now comes directly from state — reconstruction loop removed
theta_v_history = np.arctan2(vy, vx)

# Convert to degrees for readability
theta_b_deg = np.degrees(theta_b_sol)
theta_v_deg = np.degrees(theta_v_history)

# Plot 1 - Altitude vs Time
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='Altitude', color='blue')
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    if pn == 2:
        label_height = py - 400
    elif pn == 3:
        label_height = py + 200
    elif pn == 4:
        label_height = py - 600
    else:
        label_height = py + 200
    plt.annotate(f'Phase {pn}',
                xy=(pt, py),
                xytext=(pt + 20, label_height),
                fontsize=8,
                color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Altitude vs Time — Case 2A")
plt.grid()
plt.legend()
save_and_show("01_altitude_vs_time.png")

# Plot 2 - Trajectory
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Trajectory', color='blue')
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    x_phase = x[np.argmin(np.abs(t - pt))]
    plt.axvline(x=x_phase, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(x_phase, py),
                xytext=(x_phase + 500, py + 100),
                fontsize=8,
                color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
plt.xlabel("Downrange Distance (m)")
plt.ylabel("Altitude (m)")
plt.title("Trajectory — Case 2A")
plt.grid()
plt.legend()
save_and_show("02_trajectory.png")

# Plot 3 - Velocity Components
plt.figure(figsize=(10, 6))
plt.plot(t, vx, label='vx (horizontal)', color='green')
plt.plot(t, vy, label='vy (vertical)', color='orange')
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, 0),
                xytext=(pt + 5, 10),
                fontsize=8,
                color='red')
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Velocity Components vs Time — Case 2A")
plt.grid()
plt.legend()
save_and_show("03_velocity_components.png")

# Plot 4 - Total Speed
plt.figure(figsize=(10, 6))
plt.plot(t, v_mag_sol, label='Total Speed', color='purple')
plt.axhline(y=75, color='red', linestyle='--',
            label='Min cruise speed (~75 m/s)', alpha=0.7)
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, 80),
                xytext=(pt + 5, 85),
                fontsize=8,
                color='gray')
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title("Total Speed vs Time — Case 2A")
plt.grid()
plt.legend()
save_and_show("04_total_speed.png")

# Plot 5 - Theta comparison
# theta_b_deg now from integrated state — shows smooth rate-limited transitions
# theta_v_deg recomputed from velocity components as before
plt.figure(figsize=(10, 6))
plt.plot(t, theta_b_deg, label='theta_b (body angle)', color='blue')
plt.plot(t, theta_v_deg, label='theta_v (velocity angle)', color='orange')
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, 0),
                xytext=(pt + 5, 10),
                fontsize=8,
                color='red')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Angle (degrees)")
plt.title("Theta Body vs Theta Velocity — Case 2A")
plt.grid()
plt.legend()
save_and_show("05_theta_comparison.png")

# ============================================================
# Plot 6 – Linear Momentum Components (p_x, p_y)
# ============================================================
# Compute mass at each time step (same as in simulation)
mass_sol = m_v + m_fuel_sol

# Momentum components (kg·m/s)
p_x = mass_sol * vx
p_y = mass_sol * vy

plt.figure(figsize=(10, 6))
plt.plot(t, p_x, label='$p_x$ (horizontal momentum)', color='green')
plt.plot(t, p_y, label='$p_y$ (vertical momentum)', color='orange')

# Phase transition lines and annotations
for i, (pt, py_alt, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    # Place annotation near the bottom of the y-axis range
    y_min, y_max = plt.ylim()
    plt.annotate(f'Phase {pn}',
                 xy=(pt, y_min + 0.05*(y_max - y_min)),
                 xytext=(pt + 5, y_min + 0.1*(y_max - y_min)),
                 fontsize=8,
                 color='red')

plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel("Time (s)")
plt.ylabel("Momentum (kg·m/s)")
plt.title("Linear Momentum Components vs Time — Case 2A")
plt.grid()
plt.legend()
save_and_show("06_momentum.png")

# Run summary
summary_path = os.path.join(output_dir, "run_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"Case 2A Simulation Run\n")
    f.write(f"Timestamp: {run_timestamp}\n")
    f.write(f"{'='*40}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  m_v        = {m_v} kg\n")
    f.write(f"  m_f        = {m_f} kg\n")
    f.write(f"  mm         = {mm} kg/s\n")
    f.write(f"  ve         = {ve} m/s\n")
    f.write(f"  F0         = {F0:.0f} N\n")
    f.write(f"  target     = {target} m\n")
    f.write(f"  vx0        = {vx0} m/s\n")
    f.write(f"  vy0        = {vy0} m/s\n")
    f.write(f"  CL         = {CL}\n")
    f.write(f"  Cd         = {Cd}\n")
    f.write(f"  theta_b0   = {np.degrees(state0[5]):.1f} deg\n")
    f.write(f"Phase Transitions:\n")
    for entry in phase_log:
        pn, pt, py, pv = entry
        f.write(f"  Phase {pn}: t={pt:.1f}s  y={py:.1f}m  v={pv:.1f}m/s\n")
    f.write(f"\nKey Results:\n")
    f.write(f"  Max altitude   = {max(y):.1f} m\n")
    f.write(f"  Max speed      = {max(v_mag_sol):.1f} m/s\n")
    f.write(f"  Max range      = {max(x):.1f} m\n")
    f.write(f"  Flight time    = {t[-1]:.1f} s\n")
    f.write(f"  Burn time      = {m_f/mm:.1f} s\n")
    f.write(f"  Final theta_b  = {theta_b_deg[-1]:.1f} deg\n")
    f.write(f"  Final theta_v  = {theta_v_deg[-1]:.1f} deg\n")

print(f"Run summary saved to: {summary_path}")
