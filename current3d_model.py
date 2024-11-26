import numpy as np
import plotly.graph_objects as go

# Physical parameters
k = 1e-12  # Permeability in m^2
mu = 1e-3  # Dynamic viscosity in Pa.s
q_in = 1e-9  # Inflow rate at the point source in m^3/s
R = 0.005  # Cylinder radius in meters
L = 0.05  # Cylinder length in meters (can be thought of as height)
delta_P = 1000  # Axial pressure difference in Pa

# Computational grid parameters
nr = 20  # Number of radial grid points
n_theta = 20  # Number of angular grid points
nz = 50  # Number of axial grid points

# Grid generation
r = np.linspace(1e-6, R, nr)
theta = np.linspace(0, 2 * np.pi, n_theta)
z = np.linspace(1e-6, L, nz)

R_grid, Theta_grid, Z_grid = np.meshgrid(r, theta, z, indexing="ij")

# Convert to Cartesian coordinates for plotting
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)
Z = Z_grid

# radial and axiel Q
Q_r = (k * 2 * np.pi * r[:, None, None] * delta_P) / (mu * R)
Q_z = (k * np.pi * R**2 * delta_P) / (mu * L)

# Corrected Potential Function Φ
dPhi_dr = -(Q_r / (4 * np.pi)) * (r[:, None, None] / np.sqrt(r[:, None, None] ** 2 + Z_grid**2))
dPhi_dz = -(Q_z / (4 * np.pi)) * (Z_grid / np.sqrt(R_grid**2 + Z_grid**2))
dPhi_dtheta = 0  # No variation in theta due to symmetry

# Velocity components using Darcy's law
u_r = dPhi_dr  # Radial velocity
u_theta = dPhi_dtheta  # Angular velocity (zero)
u_z = dPhi_dz  # Axial velocity

# Convert velocity components to Cartesian coordinates
u_x = u_r * np.cos(Theta_grid)
u_y = u_r * np.sin(Theta_grid)
# u_z remains the same

# Prepare data for plotting
# Subsample data for better performance
step = 2
X_plot = X[::step, ::step, ::step].flatten()
Y_plot = Y[::step, ::step, ::step].flatten()
Z_plot = Z[::step, ::step, ::step].flatten()
u_x_plot = u_x[::step, ::step, ::step].flatten()
u_y_plot = u_y[::step, ::step, ::step].flatten()
u_z_plot = u_z[::step, ::step, ::step].flatten()

# Normalize vectors for visualization
vector_magnitude = np.sqrt(u_x_plot**2 + u_y_plot**2 + u_z_plot**2)
u_x_norm = u_x_plot / vector_magnitude
u_y_norm = u_y_plot / vector_magnitude
u_z_norm = u_z_plot / vector_magnitude

# Handle any NaN values due to division by zero
u_x_norm = np.nan_to_num(u_x_norm)
u_y_norm = np.nan_to_num(u_y_norm)
u_z_norm = np.nan_to_num(u_z_norm)
vector_magnitude = np.nan_to_num(vector_magnitude)


# Create 3D quiver plot using Plotly
fig = go.Figure(
    data=go.Cone(
        x=X_plot,
        y=Y_plot,
        z=Z_plot,
        u=u_x_norm,
        v=u_y_norm,
        w=u_z_norm,
        colorscale="Viridis",
        sizemode="scaled",
        sizeref=0.5,
        showscale=True,
        colorbar=dict(title="Velocity magnitude"),
    )
)

# Add cylinder surface for context
# Create cylinder surface data
theta_surface = np.linspace(0, 2 * np.pi, 50)
z_surface = np.linspace(0, L, 50)
theta_surface, z_surface = np.meshgrid(theta_surface, z_surface)
x_surface = R * np.cos(theta_surface)
y_surface = R * np.sin(theta_surface)
z_surface = z_surface

# Add cylinder surface to the figure
fig.add_trace(
    go.Surface(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        showscale=False,
        opacity=0.3,
        name="Cylinder Surface",
    )
)

# Update layout
fig.update_layout(
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",
    ),
    title="3D Velocity Field in the Cylinder",
)

# Show the figure
fig.show()
