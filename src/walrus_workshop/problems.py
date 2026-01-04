import numpy as np

def init_sod(nx, ny):
    """
    Standard Sod Shock Tube aligned with X-axis.
    Left: High Pressure/Density. Right: Low Pressure/Density.
    """
    Gamma = 1.4
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Initialize arrays
    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx))
    
    # Setup the shock interface at x = 0.5
    mask = X > 0.5
    
    # Left State (High)
    rho[~mask] = 1.0
    p[~mask]   = 1.0
    
    # Right State (Low)
    rho[mask] = 0.125
    p[mask]   = 0.1
    
    return rho, u, v, p, Gamma

def init_sedov(nx, ny):
    """
    Injection of Energy (Pressure) in the center cell.
    Background: roughly vacuum.
    """
    Gamma = 1.4
    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx)) * 1e-5  # Near-vacuum background pressure
    
    # Inject energy in the center region (r < radius)
    r_energy = 0.05
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Distance from center (0.5, 0.5)
    r = np.sqrt((X - 0.5)**2 + (Y - 0.5)**2)
    
    # Add high pressure spike in center
    p[r < r_energy] = 1.0 / (np.pi * r_energy**2) # Normalize energy
    
    return rho, u, v, p, Gamma

def init_kelvin_helmholtz(nx, ny):
    """
    Shear flow with a central strip moving right, background moving left.
    """
    Gamma = 1.4
    rho = np.ones((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.ones((ny, nx)) * 2.5
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Define the strip: |y - 0.5| < 0.25
    strip_mask = np.abs(Y - 0.5) < 0.25
    
    # Density: 2.0 inside strip, 1.0 outside
    rho[strip_mask] = 2.0
    rho[~strip_mask] = 1.0
    
    # Velocity: 0.5 inside strip, -0.5 outside
    u[strip_mask] = 0.5
    u[~strip_mask] = -0.5
    
    # Add a small sin-wave perturbation to v to trigger instability
    # w0 is perturbation amplitude (e.g. 0.1)
    w0 = 0.1
    v += w0 * np.sin(4 * np.pi * X)
    
    return rho, u, v, p, Gamma

def init_rayleigh_taylor(nx, ny):
    """
    Heavy fluid on top, light on bottom.
    Requires gravity g in -y direction (e.g., g=-0.1).
    """
    Gamma = 1.4
    rho = np.zeros((ny, nx))
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    # Fluid interfaces
    y_interface = 0.5
    rho[Y > y_interface] = 2.0  # Heavy fluid on top
    rho[Y <= y_interface] = 1.0 # Light fluid on bottom
    
    # Hydrostatic pressure gradient P = P0 + rho * g * y
    # Assuming P0 at y=1 is large enough to keep P > 0
    # Let's say P(y=1) = 2.5, g = -0.1
    g = -0.1
    P_top = 2.5
    
    # Integration from top down for hydrostatic balance is tricky in discrete,
    # but a simple linear approx often works for initialization:
    p = P_top + rho * abs(g) * (1.0 - Y)
    
    # Velocity perturbation to trigger mixing
    v += 0.01 * (1 + np.cos(4 * np.pi * X)) * (1 + np.cos(3 * np.pi * Y))
    
    return rho, u, v, p, Gamma