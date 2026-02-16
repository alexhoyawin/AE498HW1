import numpy as np
from scipy.optimize import differential_evolution

# Constants
AU_KM = 149597870.7

def keplerian_to_position(a, e, i_deg, w_deg, O_deg, f_rad):
    """
    Converts Keplerian elements to a heliocentric position vector.
    Handles both Elliptic and Hyperbolic cases.
    """
    # Convert angles to radians
    i = np.radians(i_deg)
    w = np.radians(w_deg)
    O = np.radians(O_deg)

    # 1. Position in Perifocal Frame
    # Formula: r = p / (1 + e*cos(f))
    # p = a(1-e^2) applies to both ellipse and hyperbola
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(f_rad))

    # Position vector in perifocal coordinates (P, Q, W)
    r_pqw = np.array([
        r * np.cos(f_rad),
        r * np.sin(f_rad),
        0
    ])

    # 2. Rotation Matrices (Perifocal -> Heliocentric)
    # R = Rz(O) * Rx(i) * Rz(w)
    
    # Rz(w)
    cw, sw = np.cos(w), np.sin(w)
    R_w = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    
    # Rx(i)
    ci, si = np.cos(i), np.sin(i)
    R_i = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    
    # Rz(O)
    cO, sO = np.cos(O), np.sin(O)
    R_O = np.array([[cO, -sO, 0], [sO, cO, 0], [0, 0, 1]])

    # Combined Rotation
    r_vec = R_O @ (R_i @ (R_w @ r_pqw))
    return r_vec

def distance_func(x, body1, body2):
    """
    Objective function: Euclidean distance between two bodies at true anomalies x[0] and x[1].
    """
    f1, f2 = x
    r1 = keplerian_to_position(*body1, f1)
    r2 = keplerian_to_position(*body2, f2)
    return np.linalg.norm(r1 - r2)

# --- Orbital Elements from Table 1 ---
# Format: (a [km], e, i [deg], w [deg], O [deg])
earth = (1.4765067e8, 9.1669995e-3, 4.2422693e-3, 66.4375167, 14.760836)
apophis = (1.3793939e8, 1.9097084e-1, 3.3356539, 129.19949, 203.81969)
yr4_2024 = (3.7680703e8, 6.6164147e-1, 3.4001497, 134.29905, 271.47904)
atlas_31 = (-3.9552667e7, 6.1469268, 175.12507, 128.17255, 322.28906)

# --- Define Bounds ---
# Ellipses: [0, 2pi]
bounds_ellipse = (0, 2 * np.pi)

# Hyperbola (31/ATLAS): e = 6.14. 
# Asymptotes exist where cos(f) = -1/e.
# We must constrain f strictly between the asymptotes to avoid singularities.
f_lim_atlas = np.arccos(-1 / atlas_31[1]) - 0.01  # Subtract small buffer
bounds_hyperbola = (-f_lim_atlas, f_lim_atlas)

# List of cases to solve
cases = [
    ("Earth vs Apophis", earth, apophis, [bounds_ellipse, bounds_ellipse]),
    ("Earth vs 2024 YR4", earth, yr4_2024, [bounds_ellipse, bounds_ellipse]),
    ("Apophis vs 2024 YR4", apophis, yr4_2024, [bounds_ellipse, bounds_ellipse]),
    ("Earth vs 31/ATLAS", earth, atlas_31, [bounds_ellipse, bounds_hyperbola])
]

print(f"{'Case':<25} | {'Dist (km)':<15} | {'Dist (AU)':<10} | {'f1 (deg)':<10} | {'f2 (deg)':<10}")
print("-" * 90)

for name, b1, b2, bounds in cases:
    # Use differential_evolution to find global minimum (MOID often has local minima)
    result = differential_evolution(distance_func, bounds, args=(b1, b2), strategy='best1bin', tol=1e-7)
    
    dist_km = result.fun
    dist_au = dist_km / AU_KM
    
    # Convert f1 to degrees and normalize to 0-360
    f1_deg = np.degrees(result.x[0]) % 360
    
    # Handle f2 (Hyperbola vs Ellipse display)
    f2_val = result.x[1]
    if b2[1] > 1: # if hyperbolic (e > 1)
        f2_deg = np.degrees(f2_val) # Keep range -90 to 90 etc
    else:
        f2_deg = np.degrees(f2_val) % 360

    print(f"{name:<25} | {dist_km:,.1f}     | {dist_au:.6f}   | {f1_deg:8.2f}   | {f2_deg:8.2f}")