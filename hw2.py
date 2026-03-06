import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# --- 1. Constants & Setup ---
# Standard gravitational parameter of the Sun (in AU^3 / day^2)
mu_sun = 0.01720209895**2 

# Earth Orbital Elements
q_E = 0.978969676  # Perihelion (AU)
e_E = 0.0183889630 # Eccentricity
i_E = 7.55697209e-05 # Inclination (rad)
w_E = 5.34322623   # Argument of periapsis (rad)
W_E = 2.63906227   # Longitude of ascending node (rad)
M0_E = 1.88972281  # Initial Mean Anomaly (rad)

# FI2026 Orbital Elements
q_A = 0.74161536   # Perihelion (AU)
e_A = 0.19293483   # Eccentricity
i_A = 0.05877868   # Inclination (rad)
w_A = 2.22509624   # Argument of periapsis (rad)
W_A = 3.55784498   # Longitude of ascending node (rad)
M0_A = 3.64624167  # Initial Mean Anomaly (rad)

# Calculate Semi-major axes (a = q / (1 - e))
a_E = q_E / (1 - e_E)
a_A = q_A / (1 - e_A)

# Calculate Mean Motions (n = sqrt(mu / a^3))
n_E = np.sqrt(mu_sun / a_E**3)
n_A = np.sqrt(mu_sun / a_A**3)

# --- 2. Core Functions ---
def solve_kepler(M, e, tol=1e-8):
    """Solves Kepler's Equation M = E - e*sin(E) for Eccentric Anomaly (E)."""
    E = M
    for _ in range(100):  # Added a safety limit on iterations
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if abs(delta_E) < tol:
            break
    return E

def get_position(a, e, i, w, W, M):
    """Calculates the 3D heliocentric position vector [x, y, z] for a given Mean Anomaly."""
    E = solve_kepler(M, e)
    
    # True anomaly
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # Radius
    r = a * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in the orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    
    # Transformation matrices to Heliocentric Ecliptic coordinate frame
    cw, sw = np.cos(w), np.sin(w)
    cW, sW = np.cos(W), np.sin(W)
    ci, si = np.cos(i), np.sin(i)
    
    px = cW*cw - sW*sw*ci
    py = sW*cw + cW*sw*ci
    pz = sw*si
    
    qx = -cW*sw - sW*cw*ci
    qy = -sW*sw + cW*cw*ci
    qz = cw*si
    
    x = x_orb * px + y_orb * qx
    y = x_orb * py + y_orb * qy
    z = x_orb * pz + y_orb * qz
    
    return np.array([x, y, z])

def distance_at_time(t):
    """Objective function: Returns the distance between Earth and FI2026 at time t (in days)."""
    M_E_t = M0_E + n_E * t
    M_A_t = M0_A + n_A * t
    
    pos_E = get_position(a_E, e_E, i_E, w_E, W_E, M_E_t)
    pos_A = get_position(a_A, e_A, i_A, w_A, W_A, M_A_t)
    
    return np.linalg.norm(pos_E - pos_A)

# --- 3. Global Minimum Search (Grid Search) ---
print("Running grid search to find the global minimum... Please wait.")
# Check 10,000 points over 5,000 days (half-day resolution)
times_array = np.linspace(0, 5000, 10000) 
distances_array = [distance_at_time(t) for t in times_array]

# Find the lowest point in the grid
min_index = np.argmin(distances_array)
approx_impact_time = times_array[min_index]
print(f"Grid search found a deep plunge near day {approx_impact_time:.2f}")

# --- 4. Refined Optimization ---
print("Refining exact impact time...")
# Automatically set tight bounds (+/- 10 days) around the plunge we just found
lower_bound = max(0, approx_impact_time - 10)
upper_bound = approx_impact_time + 10

result = minimize_scalar(distance_at_time, bounds=(lower_bound, upper_bound), method='bounded')

time_to_impact = result.x
miss_distance_AU = result.fun
miss_distance_km = miss_distance_AU * 149597870.7

print("\n" + "="*40)
print(f"Exact Time to Impact: {time_to_impact:.4f} days")
print(f"Minimum distance at impact: {miss_distance_km:.2f} km")
print("="*40 + "\n")

# --- 5. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(times_array, distances_array, label='Distance between Earth & FI2026', color='blue')
plt.axvline(x=time_to_impact, color='red', linestyle='--', 
            label=f'Impact Event (Day {time_to_impact:.1f})')
plt.xlabel('Time (days)')
plt.ylabel('Distance (AU)')
plt.title('Earth-FI2026 Distance vs. Time')
plt.legend()
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# --- New Function: Calculate both Position and Velocity ---
def get_state(a, e, i, w, W, M, mu=mu_sun):
    """Calculates 3D position and velocity vectors."""
    # 1. Solve Kepler's Equation
    E = M
    for _ in range(100):
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if abs(delta_E) < 1e-8:
            break
            
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # 2. Perifocal Distance and Velocity
    p = a * (1 - e**2)
    h = np.sqrt(mu * p)
    r = p / (1 + e * np.cos(nu))
    
    # Perifocal position vector
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0])
    
    # Perifocal velocity vector
    v_pqw = np.array([-mu/h * np.sin(nu), mu/h * (e + np.cos(nu)), 0])
    
    # 3. Rotation Matrix to Heliocentric Frame
    cw, sw = np.cos(w), np.sin(w)
    cW, sW = np.cos(W), np.sin(W)
    ci, si = np.cos(i), np.sin(i)
    
    R = np.array([
        [cW*cw - sW*sw*ci, -cW*sw - sW*cw*ci,  sW*si],
        [sW*cw + cW*sw*ci, -sW*sw + cW*cw*ci, -cW*si],
        [sw*si,             cw*si,             ci]
    ])
    
    # Apply rotation
    r_helio = R @ r_pqw
    v_helio = R @ v_pqw
    
    return r_helio, v_helio

# --- B-Plane Construction ---
# Use the exact time_to_impact you found in Step 1
M_E_impact = M0_E + n_E * time_to_impact
M_A_impact = M0_A + n_A * time_to_impact

r_E, v_E = get_state(a_E, e_E, i_E, w_E, W_E, M_E_impact)
r_A, v_A = get_state(a_A, e_A, i_A, w_A, W_A, M_A_impact)

# 1. Relative Position and Velocity
dr = r_A - r_E
U_vec = v_A - v_E

# 2. Define b-plane Unit Vectors
eta_hat = U_vec / np.linalg.norm(U_vec)
k_hat = np.array([0, 0, 1]) # Ecliptic North Pole

xi_vec = np.cross(eta_hat, k_hat)
xi_hat = xi_vec / np.linalg.norm(xi_vec)

zeta_hat = np.cross(xi_hat, eta_hat)

# 3. Project Relative Position onto b-plane (in AU)
xi_AU = np.dot(dr, xi_hat)
zeta_AU = np.dot(dr, zeta_hat)

# Convert AU to Earth Radii (1 AU = ~23454.8 Earth Radii)
AU_to_RE = 149597870.7 / 6371.0
xi_RE = xi_AU * AU_to_RE
zeta_RE = zeta_AU * AU_to_RE

print(f"\n--- B-Plane Coordinates ---")
print(f"Xi (ξ): {xi_RE:.4f} Earth Radii")
print(f"Zeta (ζ): {zeta_RE:.4f} Earth Radii")

# --- Plotting the B-Plane ---
fig, ax = plt.subplots(figsize=(6,6))

# Draw Earth as a circle at the origin (Radius = 1 R_E)
earth = plt.Circle((0, 0), 1.0, color='blue', alpha=0.5, label='Earth')
ax.add_patch(earth)

# Plot the asteroid
ax.plot(xi_RE, zeta_RE, 'ro', markersize=8, label='FI2026 Impact Point')

# Formatting the plot
ax.set_aspect('equal', 'box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('$\\xi$ (Earth Radii)')
ax.set_ylabel('$\\zeta$ (Earth Radii)')
ax.set_title('Öpik b-plane Encounter Visualization')
ax.grid(True, linestyle='--')
ax.legend()

plt.show()

from scipy.integrate import solve_ivp

# --- Step 3: Continuous Deflection Setup ---

def gpe_derivatives(t, state, a_T):
    """Differential equations for the orbital elements under continuous transverse thrust."""
    a, e, i, w, W, M = state
    
    # Current mean motion
    n = np.sqrt(mu_sun / a**3)
    
    # Solve Kepler's Equation for current true anomaly
    E = solve_kepler(M, e)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # Intermediate orbital parameters
    p = a * (1 - e**2)
    h = np.sqrt(mu_sun * p)
    r = p / (1 + e * np.cos(nu))
    b = a * np.sqrt(1 - e**2)
    
    # Gauss Planetary Equations (Transverse Acceleration Only)
    da_dt = (2 * a**2 / h) * (p / r) * a_T
    de_dt = (1 / h) * ((p + r) * np.cos(nu) + r * e) * a_T
    di_dt = 0.0
    dW_dt = 0.0
    dw_dt = (1 / (h * e)) * (p + r) * np.sin(nu) * a_T
    dM_dt = n - (b / (h * e)) * (p + r) * np.sin(nu) * a_T
    
    return [da_dt, de_dt, di_dt, dw_dt, dW_dt, dM_dt]

def simulate_deflection(a_T_test):
    """Integrates the orbit with a test acceleration and returns the new b-plane miss distance."""
    initial_state_A = [a_A, e_A, i_A, w_A, W_A, M0_A]
    
    # Integrate from t=0 to the exact time of impact
    # Use the `time_to_impact` variable you found in Step 1 (e.g. 360.0 days)
    sol = solve_ivp(gpe_derivatives, [0, time_to_impact], initial_state_A, 
                    args=(a_T_test,), method='RK45', rtol=1e-9, atol=1e-9)
    
    # Get the newly deflected orbital elements at the time of impact
    deflected_state = sol.y[:, -1]
    a_def, e_def, i_def, w_def, W_def, M_def = deflected_state
    
    # Calculate new 3D positions at impact time
    M_E_impact = M0_E + n_E * time_to_impact # Earth is unperturbed
    r_E_new, v_E_new = get_state(a_E, e_E, i_E, w_E, W_E, M_E_impact)
    r_A_new, v_A_new = get_state(a_def, e_def, i_def, w_def, W_def, M_def)
    
    # --- Reconstruct B-Plane ---
    dr = r_A_new - r_E_new
    U_vec = v_A_new - v_E_new
    
    eta_hat = U_vec / np.linalg.norm(U_vec)
    k_hat = np.array([0, 0, 1])
    xi_vec = np.cross(eta_hat, k_hat)
    xi_hat = xi_vec / np.linalg.norm(xi_vec)
    zeta_hat = np.cross(xi_hat, eta_hat)
    
    xi_AU = np.dot(dr, xi_hat)
    zeta_AU = np.dot(dr, zeta_hat)
    
    # Convert AU to Earth Radii
    AU_to_RE = 149597870.7 / 6371.0
    xi_RE = xi_AU * AU_to_RE
    zeta_RE = zeta_AU * AU_to_RE
    
    miss_distance = np.sqrt(xi_RE**2 + zeta_RE**2)
    return miss_distance, xi_RE, zeta_RE

# --- Iteratively find the required acceleration ---
print("\nCalculating required continuous acceleration for > 4 R_E deflection...")

# Start with a tiny acceleration in AU/day^2
test_accel = 1e-12 
miss_dist = 0

while miss_dist < 4.0:
    miss_dist, xi_final, zeta_final = simulate_deflection(test_accel)
    if miss_dist < 4.0:
        test_accel += 5e-13 # Increment and try again

# Calculate total Delta-V (acceleration * time)
delta_v_AU_day = test_accel * time_to_impact
# Convert Delta-V to meters per second (1 AU/day = 1731456.8 m/s)
delta_v_m_s = delta_v_AU_day * 1731456.8368 

print(f"Required Acceleration: {test_accel:.2e} AU/day^2")
print(f"Final Miss Distance: {miss_dist:.4f} Earth Radii")
print(f"Total Required Delta-V: {delta_v_m_s:.4f} m/s")

# --- Step 4: Kinetic Impactor (Difference Equations) ---

print("\nCalculating required instantaneous Delta-V (Kinetic Impactor) at t=0...")

def simulate_kinetic_impact(delta_V_T_AU_day):
    """Applies instantaneous Delta-V using difference equations and calculates miss distance."""
    # 1. State at t=0 (Initial elements)
    a, e, i, w, W, M = a_A, e_A, i_A, w_A, W_A, M0_A
    
    # Calculate true anomaly at t=0
    E = solve_kepler(M, e)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    # Intermediate parameters
    p = a * (1 - e**2)
    h = np.sqrt(mu_sun * p)
    r = p / (1 + e * np.cos(nu))
    b = a * np.sqrt(1 - e**2)
    
    # 2. Difference Equations (Transverse Delta-V)
    delta_a = (2 * a**2 / h) * (p / r) * delta_V_T_AU_day
    delta_e = (1 / h) * ((p + r) * np.cos(nu) + r * e) * delta_V_T_AU_day
    delta_w = (1 / (h * e)) * (p + r) * np.sin(nu) * delta_V_T_AU_day
    delta_M0 = - (b / (h * e)) * (p + r) * np.sin(nu) * delta_V_T_AU_day
    
    # 3. Apply the changes to get the NEW orbital elements
    a_new = a + delta_a
    e_new = e + delta_e
    w_new = w + delta_w
    M0_new = M + delta_M0
    
    # 4. Propagate the NEW orbit to the impact time
    n_new = np.sqrt(mu_sun / a_new**3)
    M_A_impact_new = M0_new + n_new * time_to_impact
    
    # Earth remains unchanged
    M_E_impact = M0_E + n_E * time_to_impact
    r_E_new, v_E_new = get_state(a_E, e_E, i_E, w_E, W_E, M_E_impact)
    r_A_new, v_A_new = get_state(a_new, e_new, i_A, w_new, W_A, M_A_impact_new) # i and W don't change
    
    # 5. Calculate new B-plane coordinates
    dr = r_A_new - r_E_new
    U_vec = v_A_new - v_E_new
    
    eta_hat = U_vec / np.linalg.norm(U_vec)
    k_hat = np.array([0, 0, 1])
    xi_vec = np.cross(eta_hat, k_hat)
    xi_hat = xi_vec / np.linalg.norm(xi_vec)
    zeta_hat = np.cross(xi_hat, eta_hat)
    
    xi_AU = np.dot(dr, xi_hat)
    zeta_AU = np.dot(dr, zeta_hat)
    
    AU_to_RE = 149597870.7 / 6371.0
    xi_RE = xi_AU * AU_to_RE
    zeta_RE = zeta_AU * AU_to_RE
    
    miss_distance = np.sqrt(xi_RE**2 + zeta_RE**2)
    return miss_distance, xi_RE, zeta_RE

# --- Iteratively find the required Delta-V ---
# Start with a tiny Delta-V in AU/day
test_dV = 1e-9 
miss_dist_kinetic = 0

while miss_dist_kinetic < 4.0:
    miss_dist_kinetic, xi_k, zeta_k = simulate_kinetic_impact(test_dV)
    if miss_dist_kinetic < 4.0:
        test_dV += 1e-9 # Increment and try again

# Convert Delta-V to meters per second (1 AU/day = 1731456.8 m/s)
delta_v_kinetic_m_s = test_dV * 1731456.8368 

print(f"Required Instantaneous Delta-V: {delta_v_kinetic_m_s:.4f} m/s")
print(f"Final Miss Distance: {miss_dist_kinetic:.4f} Earth Radii")

# --- Step 5: Evaluating Directional Effectiveness ---

print("\nEvaluating Radial, Transverse, and Normal directions...")

# The successful Delta-V magnitude from Step 4
test_dV_magnitude = test_dV # ~ 2.37e-7 AU/day

def simulate_directional_impact(dV_R, dV_T, dV_N):
    """Applies instantaneous Delta-V in R, T, N directions."""
    a, e, i, w, W, M = a_A, e_A, i_A, w_A, W_A, M0_A
    
    E = solve_kepler(M, e)
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
    
    p = a * (1 - e**2)
    h = np.sqrt(mu_sun * p)
    r = p / (1 + e * np.cos(nu))
    b = a * np.sqrt(1 - e**2)
    
    # Gauss Planetary Equations in Difference Form for R, T, N
    delta_a = (2 * a**2 / h) * (e * np.sin(nu) * dV_R + (p / r) * dV_T)
    
    delta_e = (1 / h) * (p * np.sin(nu) * dV_R + ((p + r) * np.cos(nu) + r * e) * dV_T)
    
    delta_i = (r * np.cos(nu + w) / h) * dV_N
    
    delta_W = (r * np.sin(nu + w) / (h * np.sin(i))) * dV_N
    
    delta_w = (1 / (h * e)) * (-p * np.cos(nu) * dV_R + (p + r) * np.sin(nu) * dV_T) \
              - (r * np.sin(nu + w) * np.cos(i) / (h * np.sin(i))) * dV_N
              
    delta_M0 = (b / (h * e)) * ((p * np.cos(nu) - 2 * r * e) * dV_R - (p + r) * np.sin(nu) * dV_T)
    
    # Apply changes
    a_new = a + delta_a
    e_new = e + delta_e
    i_new = i + delta_i
    W_new = W + delta_W
    w_new = w + delta_w
    M0_new = M + delta_M0
    
    # Propagate
    n_new = np.sqrt(mu_sun / a_new**3)
    M_A_impact_new = M0_new + n_new * time_to_impact
    
    M_E_impact = M0_E + n_E * time_to_impact
    r_E_new, v_E_new = get_state(a_E, e_E, i_E, w_E, W_E, M_E_impact)
    r_A_new, v_A_new = get_state(a_new, e_new, i_new, w_new, W_new, M_A_impact_new)
    
    # B-Plane
    dr = r_A_new - r_E_new
    U_vec = v_A_new - v_E_new
    
    eta_hat = U_vec / np.linalg.norm(U_vec)
    k_hat = np.array([0, 0, 1])
    xi_vec = np.cross(eta_hat, k_hat)
    xi_hat = xi_vec / np.linalg.norm(xi_vec)
    zeta_hat = np.cross(xi_hat, eta_hat)
    
    xi_AU = np.dot(dr, xi_hat)
    zeta_AU = np.dot(dr, zeta_hat)
    
    AU_to_RE = 149597870.7 / 6371.0
    return xi_AU * AU_to_RE, zeta_AU * AU_to_RE

# Simulate the three directions using the SAME Delta-V magnitude
xi_T, zeta_T = simulate_directional_impact(0, test_dV_magnitude, 0)
xi_R, zeta_R = simulate_directional_impact(test_dV_magnitude, 0, 0)
xi_N, zeta_N = simulate_directional_impact(0, 0, test_dV_magnitude)

print(f"Transverse Miss: {np.sqrt(xi_T**2 + zeta_T**2):.2f} R_E")
print(f"Radial Miss:     {np.sqrt(xi_R**2 + zeta_R**2):.2f} R_E")
print(f"Normal Miss:     {np.sqrt(xi_N**2 + zeta_N**2):.2f} R_E")

# --- Plotting Step 5 ---
fig, ax = plt.subplots(figsize=(8,8))

# Draw Earth & safe zone
earth = plt.Circle((0, 0), 1.0, color='blue', alpha=0.5, label='Earth')
safe_zone = plt.Circle((0, 0), 4.0, color='gray', alpha=0.2, label='4 R_E Safety Threshold')
ax.add_patch(earth)
ax.add_patch(safe_zone)

# Plot Original Impact Point
ax.plot(0, 0, 'ko', markersize=6, label='Original Impact (0,0)')

# Plot Deflections
ax.plot(xi_T, zeta_T, 'go', markersize=8, label=f'Transverse Push')
ax.plot(xi_R, zeta_R, 'ro', markersize=8, label=f'Radial Push')
ax.plot(xi_N, zeta_N, 'mo', markersize=8, label=f'Normal Push')

ax.set_aspect('equal', 'box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('$\\xi$ (Earth Radii)')
ax.set_ylabel('$\\zeta$ (Earth Radii)')
ax.set_title('Directional Effectiveness (Equal $\Delta V$)')
ax.grid(True, linestyle='--')
ax.legend()

plt.show()

# --- Step 6: Valsecchi Circles (Resonant Keyholes) ---
print("\nMapping Valsecchi Circles (This may take 10-20 seconds)...")

# Earth's gravitational parameter in AU^3/day^2
mu_earth_AU_day = 8.887692e-10 

# 1. Create a dense grid on the b-plane
grid_size = 400
xi_grid = np.linspace(-5, 5, grid_size)
zeta_grid = np.linspace(-5, 5, grid_size)
XI_RE, ZETA_RE = np.meshgrid(xi_grid, zeta_grid)

# Convert grid to AU for physics calculations
AU_to_RE = 149597870.7 / 6371.0
XI_AU = XI_RE / AU_to_RE
ZETA_AU = ZETA_RE / AU_to_RE

Period_map = np.zeros_like(XI_AU)

# Unperturbed relative velocity at impact (from Step 2)
v_inf = np.linalg.norm(U_vec)
eta_hat = U_vec / v_inf

# 2. Iterate over the grid to simulate gravitational scattering
for i in range(grid_size):
    for j in range(grid_size):
        xi = XI_AU[i, j]
        zeta = ZETA_AU[i, j]
        b_mag = np.sqrt(xi**2 + zeta**2)
        
        # If it impacts Earth directly (< 1 RE), there is no post-encounter orbit
        if b_mag * AU_to_RE < 1.0:
            Period_map[i, j] = np.nan
            continue
            
        # Impact parameter unit vector
        b_vec = xi * xi_hat + zeta * zeta_hat
        b_hat = b_vec / b_mag
        
        # Deflection angle (Rutherford Scattering equation)
        gamma = 2 * np.arcsin(1 / np.sqrt(1 + (b_mag * v_inf**2 / mu_earth_AU_day)**2))
        
        # Axis of rotation (Gravity bends the trajectory TOWARDS Earth, so cross(b_hat, eta_hat))
        k_rot = np.cross(b_hat, eta_hat)
        k_rot_hat = k_rot / np.linalg.norm(k_rot)
        
        # Rodrigues Rotation Formula to bend the relative velocity vector
        K = np.array([
            [0, -k_rot_hat[2], k_rot_hat[1]],
            [k_rot_hat[2], 0, -k_rot_hat[0]],
            [-k_rot_hat[1], k_rot_hat[0], 0]
        ])
        R_mat = np.eye(3) + np.sin(gamma) * K + (1 - np.cos(gamma)) * (K @ K)
        
        U_prime = R_mat @ U_vec
        
        # Calculate New Heliocentric Velocity & Orbit
        V_prime = v_E + U_prime
        v_prime_mag = np.linalg.norm(V_prime)
        r_mag = np.linalg.norm(r_E) # Position is effectively Earth's position at encounter
        
        # Post-encounter semi-major axis (Vis-viva equation)
        a_prime = 1 / (2/r_mag - v_prime_mag**2 / mu_sun)
        
        # Post-encounter period in Earth Years (365.25 days)
        period_days = 2 * np.pi * np.sqrt(a_prime**3 / mu_sun)
        Period_map[i, j] = period_days / 365.25636

# 3. Define the Resonant Fractions (k Earth years / h Asteroid orbits)
resonances = []
for k in range(1, 20):
    for h in range(1, 20):
        ratio = k / h
        # Only care about periods close to the original (it won't jump from 0.8 to 2.0 years)
        if 0.7 < ratio < 1.3: 
            resonances.append(ratio)
resonances = sorted(list(set(resonances)))

# 4. Plot the Final Masterpiece
fig, ax = plt.subplots(figsize=(9,9))

# Draw Earth & safe zone
earth = plt.Circle((0, 0), 1.0, color='blue', alpha=0.5, label='Earth')
safe_zone = plt.Circle((0, 0), 4.0, color='gray', alpha=0.2, label='4 R_E Safety Threshold')
ax.add_patch(earth)
ax.add_patch(safe_zone)

# Draw Valsecchi Circles (Contour lines where the period map matches a resonant fraction)
CS = ax.contour(XI_RE, ZETA_RE, Period_map, levels=resonances, colors='orange', linewidths=1.0, alpha=0.6)
# Custom legend handle for the contours
ax.plot([], [], color='orange', label='Valsecchi Circles (Resonances)')

# Plot Original and Deflected Points (Using variables from your Step 5)
ax.plot(0, 0, 'ko', markersize=6, label='Original Impact (0,0)')
ax.plot(xi_T, zeta_T, 'go', markersize=8, label=f'Transverse Push (Safe)')
ax.plot(xi_R, zeta_R, 'ro', markersize=8, label=f'Radial Push (Impact)')
ax.plot(xi_N, zeta_N, 'mo', markersize=8, label=f'Normal Push (Impact)')

# Formatting
ax.set_aspect('equal', 'box')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('$\\xi$ (Earth Radii)')
ax.set_ylabel('$\\zeta$ (Earth Radii)')
ax.set_title('Öpik b-plane: Deflection vs. Valsecchi Circles')
ax.grid(True, linestyle='--')
ax.legend(loc='upper right')

plt.show()