import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

import numpy as np

def compute_deformation(u, v, dx=1.0, dy=1.0):
    """
    Compute total deformation from horizontal wind fields on a uniform grid.
    
    Parameters
    ----------
    u, v : ndarray
        Horizontal wind components, shape (..., ny, nx)
    dx, dy : float
        Grid spacing (same units as u, v for physically meaningful output,
        or just use 1.0 for relative comparisons)
    
    Returns
    -------
    total_deformation : ndarray
        Same shape as u, v
    shear_deformation : ndarray
        dv/dx + du/dy
    stretch_deformation : ndarray  
        du/dx - dv/dy
    """
    dudx = np.gradient(u, dx, axis=-1)
    dudy = np.gradient(u, dy, axis=-2)
    dvdx = np.gradient(v, dx, axis=-1)
    dvdy = np.gradient(v, dy, axis=-2)
    
    shear_def = dvdx + dudy
    stretch_def = dudx - dvdy
    
    total_deformation = np.sqrt(shear_def**2 + stretch_def**2)
    
    return total_deformation, shear_def, stretch_def

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

def compute_enstrophy_flux(u, v, dx=1.0, dy=1.0):
    enstrophy, vorticity = compute_enstrophy(u, v, dx, dy)
    return _compute_enstrophy_flux(vorticity, dx, dy)

def compute_energy_flux(u, v, dx=1.0, dy=1.0):
    energy, vorticity = compute_enstrophy(u, v, dx, dy)
    return _compute_energy_flux(vorticity, dx, dy)

def _compute_enstrophy_flux(omega, Lx, Ly):
    """
    Compute spectral enstrophy flux from a 2D vorticity field.
    
    Parameters
    ----------
    omega : ndarray (Ny, Nx)
        2D vorticity field
    Lx, Ly : float
        Domain size in x and y directions
    
    Returns
    -------
    k_bins : ndarray
        Wavenumber bin centers
    Pi_omega : ndarray
        Enstrophy flux at each wavenumber
    T_omega : ndarray
        Enstrophy transfer spectrum
    """
    Ny, Nx = omega.shape
    
    # Wavenumber arrays
    kx = 2 * np.pi * fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # FFT of vorticity
    omega_hat = fft2(omega)
    
    # Compute velocity from vorticity: u = (-∂ψ/∂y, ∂ψ/∂x)
    # where ∇²ψ = ω, so ψ_hat = -ω_hat / k²
    K_sq = KX**2 + KY**2
    K_sq[0, 0] = 1  # Avoid division by zero
    
    psi_hat = -omega_hat / K_sq
    psi_hat[0, 0] = 0  # Zero mean streamfunction
    
    u_hat = -1j * KY * psi_hat   # u = -∂ψ/∂y
    v_hat =  1j * KX * psi_hat   # v =  ∂ψ/∂x
    
    # Transform to physical space
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))
    
    # Compute vorticity gradients in spectral space
    domega_dx_hat = 1j * KX * omega_hat
    domega_dy_hat = 1j * KY * omega_hat
    
    domega_dx = np.real(ifft2(domega_dx_hat))
    domega_dy = np.real(ifft2(domega_dy_hat))
    
    # Nonlinear term: u·∇ω
    advection = u * domega_dx + v * domega_dy
    advection_hat = fft2(advection)
    
    # Enstrophy transfer: T_Ω(k) = Re[ω_hat* · (u·∇ω)_hat]
    # This is the shell-by-shell transfer
    T_omega_2d = -np.real(np.conj(omega_hat) * advection_hat)
    
    # Normalize by grid size
    T_omega_2d /= (Nx * Ny)**2
    
    # Bin into shells
    k_max = np.sqrt(2) * max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = 2 * np.pi / max(Lx, Ly)
    k_bins = np.arange(0, k_max, dk)
    n_bins = len(k_bins) - 1
    
    T_omega = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        T_omega[i] = np.sum(T_omega_2d[mask])
    
    # Compute flux: Π(k) = -∫₀ᵏ T(k') dk'
    # Positive flux = forward cascade (toward high k)
    Pi_omega = -np.cumsum(T_omega) * dk
    
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    return k_centers, Pi_omega, T_omega


def _compute_energy_flux(omega, Lx, Ly):
    """
    Compute spectral energy flux (for comparison to enstrophy flux).
    Negative flux at low k indicates inverse cascade.
    """
    Ny, Nx = omega.shape
    
    kx = 2 * np.pi * fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    omega_hat = fft2(omega)
    
    K_sq = KX**2 + KY**2
    K_sq[0, 0] = 1
    
    psi_hat = -omega_hat / K_sq
    psi_hat[0, 0] = 0
    
    u_hat = -1j * KY * psi_hat
    v_hat =  1j * KX * psi_hat
    
    u = np.real(ifft2(u_hat))
    v = np.real(ifft2(v_hat))
    
    domega_dx = np.real(ifft2(1j * KX * omega_hat))
    domega_dy = np.real(ifft2(1j * KY * omega_hat))
    
    advection = u * domega_dx + v * domega_dy
    advection_hat = fft2(advection)
    
    # Energy transfer: T_E(k) = Re[ψ_hat* · (u·∇ω)_hat]
    T_energy_2d = np.real(np.conj(psi_hat) * advection_hat)
    T_energy_2d /= (Nx * Ny)**2
    
    k_max = np.sqrt(2) * max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    dk = 2 * np.pi / max(Lx, Ly)
    k_bins = np.arange(0, k_max, dk)
    n_bins = len(k_bins) - 1
    
    T_energy = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        T_energy[i] = np.sum(T_energy_2d[mask])
    
    Pi_energy = -np.cumsum(T_energy) * dk
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    return k_centers, Pi_energy, T_energy