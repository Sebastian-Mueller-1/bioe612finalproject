import numpy as np
import plotly.graph_objects as go

# Physical parameters
k = 1e-12  # Permeability in m^2 
mu = 1e-3  # Dynamic viscosity in Pa.
q_in = 1e-9  # Inflow rate at the point source in m^3/s (volumetric flow rate entering the system)
R = 0.005  # Cylinder radius in meters
L = 0.05  # Cylinder length in meters

# Computational grid parameters
nr = 20  # Number of radial grid points
n_theta = 20  # Number of angular grid points
nz = 50  # Number of axial grid points

# Grid generation
# Generate linearly spaced points for radial, angular, and axial dimensions
r = np.linspace(1e-6, R, nr)  # Radial distances (avoid zero to prevent singularity in log)
theta = np.linspace(0, 2 * np.pi, n_theta)  # Angular coordinates (full circle)
z = np.linspace(1e-6, L, nz)  # Axial positions (avoid zero to prevent singularity)

# Create 3D mesh grid in cylindrical coordinates (r, θ, z)
R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing="ij")

# Convert cylindrical coordinates to Cartesian coordinates for plotting
# X = r*cos(θ), Y = r*sin(θ), Z = z
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)
Z = Z_grid

#Potential Function Φ (representing pressure-driven flow in porous media)
# Φ = -(Q / (4πkμ)) * log(sqrt(r^2 + z^2))
# Derived from Darcy's law in combination with potential flow theory
# Q = inflow rate, k = permeability, μ = viscosity
Q = q_in
Phi = -(Q / (4 * np.pi * k * mu)) * np.log(np.sqrt(R_grid**2 + Z_grid**2))

# pressure gradient components (partial derivatives of Φ)
# dΦ/dr = -Q/(4πkμ) * r/(r^2 + z^2), representing radial pressure gradient
dPhi_dr = -(Q / (4 * np.pi * k * mu)) * (R_grid / (R_grid**2 + Z_grid**2))

# dΦ/dz = -Q/(4πkμ) * z/(r^2 + z^2), representing axial pressure gradient
dPhi_dz = -(Q / (4 * np.pi * k * mu)) * (Z_grid / (R_grid**2 + Z_grid**2))

# dΦ/dθ = 0, no variation in angular direction due to symmetry
dPhi_dtheta = 0

# Velocity components in cylindrical coordinates using Darcy's law
# Darcy's law: velocity = -(k/μ) * gradient(pressure)
u_r = -k / mu * dPhi_dr
u_theta = -k / mu * dPhi_dtheta
u_z = -k / mu * dPhi_dz

# Convert velocity components to Cartesian coordinates for plotting
# u_x = u_r*cos(θ), u_y = u_r*sin(θ), u_z remains unchanged
u_x = u_r * np.cos(Theta_grid)
u_y = u_r * np.sin(Theta_grid)
# u_z remains as calculated in cylindrical coordinates

# Prepare data for plotting
# Subsample data to improve performance when visualizing
step = 2
X_plot = X[::step, ::step, ::step].flatten()
Y_plot = Y[::step, ::step, ::step].flatten()
Z_plot = Z[::step, ::step, ::step].flatten()
u_x_plot = u_x[::step, ::step, ::step].flatten()
u_y_plot = u_y[::step, ::step, ::step].flatten()
u_z_plot = u_z[::step, ::step, ::step].flatten()

# Normalize velocity vectors for consistent visualization (unit vector scaling)
vector_magnitude = np.sqrt(u_x_plot**2 + u_y_plot**2 + u_z_plot**2)
u_x_norm = u_x_plot / vector_magnitude
u_y_norm = u_y_plot / vector_magnitude
u_z_norm = u_z_plot / vector_magnitude

# Handle potential NaN values caused by division by zero
u_x_norm = np.nan_to_num(u_x_norm)
u_y_norm = np.nan_to_num(u_y_norm)
u_z_norm = np.nan_to_num(u_z_norm)
vector_magnitude = np.nan_to_num(vector_magnitude)

# Create 3D quiver plot using Plotly to visualize velocity field
fig = go.Figure(
    data=go.Cone(
        x=X_plot,
        y=Y_plot,
        z=Z_plot,
        u=u_x_norm,
        v=u_y_norm,
        w=u_z_norm,
        colorscale="Viridis",  # Color map for vector magnitude
        sizemode="scaled",
        sizeref=0.5,
        showscale=True,  # Show color scale bar for velocity magnitude
        colorbar=dict(title="Velocity magnitude"),  # Label for color bar
    )
)

# Add cylinder surface for context
# Create mesh for the cylinder surface (using θ and z)
theta_surface = np.linspace(0, 2 * np.pi, 50)  # Full angular range
z_surface = np.linspace(0, L, 50)  # Full axial range
theta_surface, z_surface = np.meshgrid(theta_surface, z_surface)
x_surface = R * np.cos(theta_surface)  # X-coordinates of cylinder
y_surface = R * np.sin(theta_surface)  # Y-coordinates of cylinder

# Add cylinder surface to the plot for better visualization context
fig.add_trace(
    go.Surface(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        showscale=False,  # No scale bar for the surface
        opacity=0.3,  # Semi-transparent for visual clarity
        name="Cylinder Surface",  # Legend label
    )
)

# Update layout to add axis labels and adjust aspect ratio
fig.update_layout(
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",  # Maintain data aspect ratio
    ),
    title="3D Velocity Field in the Cylinder",  # Plot title
)

# Display the figure
fig.show()
