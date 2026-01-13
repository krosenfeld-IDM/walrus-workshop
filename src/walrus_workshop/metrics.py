import numpy as np

def compute_enstrophy(u, v, dx=1.0, dy=1.0):
    """
    Compute enstrophy from 2D velocity fields.
    
    Parameters:
    -----------
    u : 2D array
        x-component of velocity
    v : 2D array
        y-component of velocity
    dx : float
        grid spacing in x-direction
    dy : float
        grid spacing in y-direction
    
    Returns:
    --------
    enstrophy : float
        Total enstrophy
    vorticity : 2D array
        Vorticity field (useful for visualization)
    """
    # Compute vorticity: ω = ∂v/∂x - ∂u/∂y
    dv_dx = np.gradient(v, dx, axis=1)  # derivative in x-direction
    du_dy = np.gradient(u, dy, axis=0)  # derivative in y-direction
    
    vorticity = dv_dx - du_dy
    
    # Enstrophy = 0.5 * integral of ω²
    # For discrete case: sum over domain and normalize by area
    enstrophy = 0.5 * np.sum(vorticity**2) * dx * dy
    
    return enstrophy, vorticity