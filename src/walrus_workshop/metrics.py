import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

import numpy as np

def coarsen_field(field, coarse_shape, method='mean'):
    """
    Coarsen a fine-grid field to a coarse grid by block averaging.
    
    Args:
        field: (ny_fine, nx_fine) array
        coarse_shape: (ny_coarse, nx_coarse)
        method: 'mean' for continuous fields, 'fraction' for binary masks
    """
    ny_f, nx_f = field.shape
    ny_c, nx_c = coarse_shape
    
    # Reshape into blocks and average
    block_y = ny_f // ny_c
    block_x = nx_f // nx_c
    
    # Trim if not evenly divisible
    trimmed = field[:ny_c * block_y, :nx_c * block_x]
    
    blocks = trimmed.reshape(ny_c, block_y, nx_c, block_x)
    return blocks.mean(axis=(1, 3))

def _compute_energy_spectrum(u, v, dx=1.0, dy=1.0):
    """
    Compute the 1D (radially averaged) kinetic energy spectrum from 2D velocity fields.

    Parameters
    ----------
    u, v : np.ndarray, shape (Ny, Nx)
        Velocity components on a uniform grid.
    dx, dy : float
        Physical domain size in x and y directions.

    Returns
    -------
    k_bins : np.ndarray
        Wavenumber bin centers.
    E_k : np.ndarray
        Energy spectrum E(k), where integral E(k) dk = 0.5 * <u^2 + v^2>.
    """
    Ny, Nx = u.shape
    Lx, Ly = Nx * dx, Ny * dy

    # Subtract the mean of the field
    u -= u.mean()
    v -= v.mean()

    # 2D FFT (no normalization — we normalize manually)
    u_hat = np.fft.fft2(u) / (Nx * Ny)
    v_hat = np.fft.fft2(v) / (Nx * Ny)

    # Energy per mode: 0.5 * (|u_hat|^2 + |v_hat|^2), scaled by domain area
    # Factor of (Nx*Ny) converts from discrete to continuous Parseval
    E_2d = 0.5 * (np.abs(u_hat) ** 2 + np.abs(v_hat) ** 2) * (Nx * Ny)

    # Wavenumber grids
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi  # rad / length
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    K_mag = np.sqrt(KX**2 + KY**2)

    # Radial binning
    dk = 2 * np.pi / max(Lx, Ly)  # bin width ~ fundamental wavenumber
    k_max = np.pi / min(dx, dy)    # Nyquist
    k_edges = np.arange(0, k_max + dk, dk)
    k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])

    E_k = np.zeros(len(k_bins))
    for i in range(len(k_bins)):
        mask = (K_mag >= k_edges[i]) & (K_mag < k_edges[i + 1])
        E_k[i] = np.sum(E_2d[mask])

    # Normalize so that sum(E_k * dk) ≈ 0.5 * mean(u^2 + v^2)
    E_k /= dk

    return k_bins, E_k

def compute_energy_spectrum(u, v, dx=1.0, dy=1.0):
    """
    Compute the 1D (radially averaged) kinetic energy spectrum from 2D velocity fields.

    Parameters
    ----------
    u, v : np.ndarray, shape (Ny, Nx)
        Velocity components on a uniform grid.
    dx, dy : float
        Physical domain size in x and y directions.

    Returns
    -------
    k_bins : np.ndarray
        Wavenumber bin centers.
    E_k : np.ndarray
        Energy spectrum E(k), where integral E(k) dk = 0.5 * <u^2 + v^2>.
    """    
    Ny, Nx = u.shape
    Lx, Ly = Nx*dx, Ny*dy
    N = Nx*Ny

    # avoid in-place modification of caller's arrays
    u0 = u - u.mean()
    v0 = v - v.mean()

    # FFT normalized so Parseval gives mean-square directly
    u_hat = np.fft.fft2(u0) / N
    v_hat = np.fft.fft2(v0) / N

    # per-mode energy; sum(E_2d) ~= 0.5*mean(u^2+v^2)
    E_2d = 0.5*(np.abs(u_hat)**2 + np.abs(v_hat)**2)

    kx = np.fft.fftfreq(Nx, d=dx) * 2*np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2*np.pi
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    # dk = min(2*np.pi/Lx, 2*np.pi/Ly)
    dk = 2*np.pi / min(Lx, Ly)   # = 2π/256 = Δky
    # k_max = np.sqrt((np.pi/dx)**2 + (np.pi/dy)**2)
    k_max = np.sqrt((np.pi/dx)**2 + (np.pi/dy)**2)
    k_edges = np.arange(0, k_max + dk, dk)
    k_bins = 0.5*(k_edges[:-1] + k_edges[1:])

    E_shell = np.zeros_like(k_bins)
    for i in range(len(k_bins)):
        mask = (K >= k_edges[i]) & (K < k_edges[i+1])
        E_shell[i] = E_2d[mask].sum()

    E_k = E_shell / dk
    return k_bins, E_k


def subgrid_stress(u_fine, v_fine, coarse_shape):
    """
    tau_ij = coarsen(u_i * u_j) - coarsen(u_i) * coarsen(u_j)
    
    This is exactly what the model must implicitly
    parameterize to make accurate predictions.
    """
    uu = coarsen_field(u_fine * u_fine, coarse_shape)
    vv = coarsen_field(v_fine * v_fine, coarse_shape)
    uv = coarsen_field(u_fine * v_fine, coarse_shape)
    
    u_c = coarsen_field(u_fine, coarse_shape)
    v_c = coarsen_field(v_fine, coarse_shape)
    
    tau_xx = uu - u_c * u_c
    tau_yy = vv - v_c * v_c
    tau_xy = uv - u_c * v_c
    
    # Subgrid kinetic energy (scalar summary)
    tke = 0.5 * (tau_xx + tau_yy)
    
    return tau_xx, tau_yy, tau_xy, tke

def compute_okubo_weiss(u, v, dx=1.0, dy=1.0):
    """
    Compute the Okubo-Weiss parameter Q = S² - ω² on a uniform grid.

    Q > 0: strain-dominated (filaments, stretching)
    Q < 0: vorticity-dominated (vortex cores)

    Args:
        u: (ny, nx) x-velocity field
        v: (ny, nx) y-velocity field
        dx: grid spacing in x
        dy: grid spacing in y

    Returns:
        Q:     (ny, nx) Okubo-Weiss parameter
        sn:    (ny, nx) normal strain  (du/dx - dv/dy)
        ss:    (ny, nx) shear strain   (dv/dx + du/dy)
        omega: (ny, nx) vorticity      (dv/dx - du/dy)
    """
    # Velocity gradients via central differences
    dudx = np.gradient(u, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    dvdx = np.gradient(v, dx, axis=1)
    dvdy = np.gradient(v, dy, axis=0)

    # Normal (stretching) strain rate
    sn = dudx - dvdy

    # Shear strain rate
    ss = dvdx + dudy

    # Vorticity
    omega = dvdx - dudy

    # Okubo-Weiss: Q = Sn² + Ss² - ω²
    Q = sn**2 + ss**2 - omega**2

    return Q, sn, ss, omega


def compute_deformation(u, v, dx=1.0, dy=1.0):
    """
    Compute total deformation from horizontal velocity fields on a uniform grid.
    
    Parameters
    ----------
    u, v : ndarray
        Horizontal velocity components, shape (..., ny, nx)
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

def _compute_enstrophy_flux(omega, dx=1.0, dy=1.0):
    """
    Compute spectral enstrophy flux from a 2D vorticity field.
    
    Parameters
    ----------
    omega : ndarray (Ny, Nx)
        2D vorticity field
    dx, dy : float
        Grid spacing in x and y directions
    
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
    Lx, Ly = Nx*dx, Ny*dy
    
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
    # k_max = np.sqrt(2) * max(np.max(np.abs(kx)), np.max(np.abs(ky)))
    # dk = 2 * np.pi / max(Lx, Ly)
    k_max = np.sqrt((np.pi/dx)**2 + (np.pi/dy)**2)
    dk = 2*np.pi / min(Lx, Ly)   # = 2π/256 = Δky    
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


def _compute_energy_flux(omega, dx=1.0, dy=1.0):
    """
    Compute spectral energy flux (for comparison to enstrophy flux).
    Negative flux at low k indicates inverse cascade.
    """
    Ny, Nx = omega.shape
    Lx, Ly = Nx*dx, Ny*dy
    
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