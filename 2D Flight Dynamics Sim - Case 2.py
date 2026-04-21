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

def save_and_show(filename, dpi=150):
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.show()
    plt.close()  # prevents memory buildup across many plots

#Non-Thrust Vectoring Case
def case2a(t, state):
    y, x, vy, vx, m_fuel = state  # renamed to m_fuel

    # Fuel and mass
    if m_fuel > 0:
        Ft = F0
        mt = m_v + m_fuel
        dm_fuel_dt = -mm
    else:
        Ft = 0
        mt = m_v
        dm_fuel_dt = 0

    v_mag = np.hypot(vx, vy)

    # Stateless theta_b calculation based on current phase list value
    # Phase advances forward only via list — never goes backward

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

    # Theta_b computed from phase[0] — forward only
    if phase[0] == 1:
        theta_b = min_angle + (max_angle - min_angle) * (y / (target / 2))
    elif phase[0] == 2:
        theta_b = max(max_angle * (1 - (y - target / 2) / (target / 2)), 0)
    elif phase[0] == 3:
        altitude_error = y - target
        if altitude_error > 25:
            theta_b = np.radians(-5)
        elif altitude_error < -25:
            theta_b = np.radians(6)
        else:
           theta_b = np.radians(-25)
    else:
        theta_b = -np.radians(15)

    # Theta_v guard
    if v_mag < 1.0 or vx <= 0:
        theta_v = theta_b
    else:
        theta_v = np.arctan2(vy, vx)

        # Theta_v guard
        if v_mag < 20.0 or vx <= 0:
            theta_v = theta_b
        else:
            theta_v = np.arctan2(vy, vx)

        # Aerodynamics — back to original simple form
        rho_a = rho * np.exp(-y / H)
        q = max(0.5 * rho_a * v_mag ** 2, 1.0)

        D = q * A * Cd
        L = q * Aw * CL

        # Force equations
        dydt = vy
        dxdt = vx
        dvydt = (Ft * np.sin(theta_b) - mt * g - D * np.sin(theta_v)+ L * np.cos(theta_v)) / mt
        dvxdt = (Ft * np.cos(theta_b)- D * np.cos(theta_v) - L * np.sin(theta_v)) / mt

        return [dydt, dxdt, dvydt, dvxdt, dm_fuel_dt]

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

# Diagontics
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

#solving v-mag over time
v_mag_sol = np.hypot(vx, vy)

# Extract phase transition points for markers
phase_times = [entry[1] for entry in phase_log]
phase_alts  = [entry[2] for entry in phase_log]
phase_nums  = [entry[0] for entry in phase_log]
phase_vmags = [entry[3] for entry in phase_log]

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

# Plot 1 - Altitude vs Time
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
plt.title("Altitude vs Time — Case 2A")
plt.grid()
plt.legend()
save_and_show("01_altitude_vs_time.png")

# Plot 2 - Trajectory
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Trajectory', color='blue')
plt.xlabel("Downrange Distance (m)")
plt.ylabel("Altitude (m)")
plt.title("Trajectory — Case 2A")
plt.grid()
plt.legend()
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    x_phase = x[np.argmin(np.abs(t - pt))]
    plt.axvline(x=x_phase, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(x_phase, py),
                xytext=(x_phase + 500, py + 100),
                fontsize=8,
                color='red',
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5))
save_and_show("02_trajectory.png")

# Plot 3 - Velocity Components
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
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, 0),
                xytext=(pt + 5, 10),
                fontsize=8,
                color='red')
save_and_show("03_velocity_components.png")

# Plot 4 - Total Speed
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
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='gray', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, 80),
                xytext=(pt + 5, 85),
                fontsize=8,
                color='gray')
save_and_show("04_total_speed.png")

# Plot 5 - Theta comparison
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
for i, (pt, py, pn) in enumerate(zip(phase_times, phase_alts, phase_nums)):
    plt.axvline(x=pt, color='red', linestyle='--', alpha=0.5)
    plt.annotate(f'Phase {pn}',
                xy=(pt, 0),
                xytext=(pt + 5, 10),
                fontsize=8,
                color='red')
save_and_show("05_theta_comparison.png")

summary_path = os.path.join(output_dir, "run_summary.txt")
with open(summary_path, 'w') as f:
    f.write(f"Case 2A Simulation Run\n")
    f.write(f"Timestamp: {run_timestamp}\n")
    f.write(f"{'='*40}\n\n")
    f.write(f"Parameters:\n")
    f.write(f"  m_v    = {m_v} kg\n")
    f.write(f"  m_f    = {m_f} kg\n")
    f.write(f"  mm     = {mm} kg/s\n")
    f.write(f"  ve     = {ve} m/s\n")
    f.write(f"  F0     = {F0:.0f} N\n")
    f.write(f"  target = {target} m\n")
    f.write(f"  vx0    = {vx0} m/s\n")
    f.write(f"  vy0    = {vy0} m/s\n")
    f.write(f"  CL     = {CL}\n")
    f.write(f"  Cd     = {Cd}\n\n")
    f.write(f"Phase Transitions:\n")
    for entry in phase_log:
        pn, pt, py, pv = entry
        f.write(f"  Phase {pn}: t={pt:.1f}s  y={py:.1f}m  v={pv:.1f}m/s\n")
    f.write(f"\nKey Results:\n")
    f.write(f"  Max altitude  = {max(y):.1f} m\n")
    f.write(f"  Max speed     = {max(v_mag_sol):.1f} m/s\n")
    f.write(f"  Max range     = {max(x):.1f} m\n")
    f.write(f"  Flight time   = {t[-1]:.1f} s\n")
    f.write(f"  Burn time     = {m_f/mm:.1f} s\n")

print(f"Run summary saved to: {summary_path}")