from flask import Flask, render_template
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Base parameters for 2D radial and 3D cylindrical analysis
mu_base = 0.01
kappa_base = 0.1
delta_p_base = 100

# Define radial velocity function for 2D radial flow (no dependency on r)
def radial_velocity(mu, kappa, delta_p):
    return (delta_p * kappa) / (2 * np.pi * mu)

# Define axial velocity function for 3D cylindrical flow
def axial_velocity(mu, kappa, delta_p, L):
    return (delta_p * kappa) / (mu * L)  # Simplified for axial flow along the cylinder's length

# Generate sensitivity data for 2D radial flow
num_points = 100
mu_values = np.linspace(0.005, 0.02, num_points)
kappa_values = np.linspace(0.05, 0.2, num_points)
delta_p_values = np.linspace(50, 150, num_points)
L_values = np.linspace(1, 20, num_points)

velocity_mu = [radial_velocity(mu, kappa_base, delta_p_base) for mu in mu_values]
velocity_kappa = [radial_velocity(mu_base, kappa, delta_p_base) for kappa in kappa_values]
velocity_delta_p = [radial_velocity(mu_base, kappa_base, delta_p) for delta_p in delta_p_values]

# Generate data for 3D axial flow
velocity_mu_3d = [axial_velocity(mu, kappa_base, delta_p_base, L_values) for mu in mu_values]
velocity_kappa_3d = [axial_velocity(mu_base, kappa, delta_p_base, L_values) for kappa in kappa_values]
velocity_delta_p_3d = [axial_velocity(mu_base, kappa_base, delta_p, L_values) for delta_p in delta_p_values]

# Create plotly figures for each parameter sensitivity
def create_plot(parameter_values, velocities, parameter_name, parameter_label):
    trace = go.Scatter(
        x=parameter_values,
        y=velocities,
        mode='lines',
        name=f'{parameter_label} Effect',  # Label for the legend
        showlegend=True  # Ensure legend is shown
    )
    layout = go.Layout(
        title=f'Sensitivity of Radial Velocity to {parameter_label}',
        xaxis=dict(title=parameter_label),
        yaxis=dict(title='Radial Velocity'),
        showlegend=True  # Ensure legend is shown in layout
    )
    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

# Create 3D surface plot for cylindrical flow
def create_3d_plot(L, mu, title):
    Z = np.array([axial_velocity(mu_val, kappa_base, delta_p_base, L_row) for mu_val, L_row in zip(mu, L)])

    trace = go.Surface(
        z=Z,
        x=L,
        y=mu,
        colorscale='Viridis',
        colorbar=dict(title='Velocity Magnitude')  # Add title to color bar
    )
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='Length L'),
            yaxis=dict(title='Viscosity μ'),
            zaxis=dict(title='Axial Velocity')
        )
    )
    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

# Create heat map for pairwise parameter combinations
def create_heat_map(param1, param2, param1_name, param2_name, title):
    Z = np.array([[axial_velocity(mu, kappa_base, delta_p_base, L) for L in param1] for mu in param2])

    trace = go.Heatmap(
        z=Z,
        x=param1,
        y=param2,
        colorscale='Viridis',
        colorbar=dict(title='Velocity Magnitude')  # Add title to color bar
    )
    layout = go.Layout(
        title=title,
        xaxis=dict(title=param1_name),
        yaxis=dict(title=param2_name)
    )
    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

@app.route('/')
def index():
    # 2D Radial flow plots
    mu_plot = create_plot(mu_values, velocity_mu, 'Viscosity (μ)', 'Viscosity (μ)')
    kappa_plot = create_plot(kappa_values, velocity_kappa, 'Permeability (κ)', 'Permeability (κ)')
    delta_p_plot = create_plot(delta_p_values, velocity_delta_p, 'Pressure Difference (ΔP)', 'Pressure Difference (ΔP)')

    # 3D Axial flow visualization for length (L) and viscosity (μ)
    L, mu = np.meshgrid(L_values, mu_values)
    velocity_3d_plot = create_3d_plot(L, mu, '3D Axial Velocity Distribution for Length (L) and Viscosity (μ)')

    # Create heat maps for all pairwise combinations
    heat_map_mu_L = create_heat_map(L_values, mu_values, 'Length (L)', 'Viscosity (μ)', 'Heat Map: L vs. μ')
    heat_map_kappa_L = create_heat_map(L_values, kappa_values, 'Length (L)', 'Permeability (κ)', 'Heat Map: L vs. κ')
    heat_map_deltaP_L = create_heat_map(L_values, delta_p_values, 'Length (L)', 'Pressure Difference (ΔP)', 'Heat Map: L vs. ΔP')
    heat_map_mu_kappa = create_heat_map(mu_values, kappa_values, 'Viscosity (μ)', 'Permeability (κ)', 'Heat Map: μ vs. κ')
    heat_map_mu_deltaP = create_heat_map(mu_values, delta_p_values, 'Viscosity (μ)', 'Pressure Difference (ΔP)', 'Heat Map: μ vs. ΔP')
    heat_map_kappa_deltaP = create_heat_map(kappa_values, delta_p_values, 'Permeability (κ)', 'Pressure Difference (ΔP)', 'Heat Map: κ vs. ΔP')

    return render_template(
        'index.html',
        mu_plot=mu_plot,
        kappa_plot=kappa_plot,
        delta_p_plot=delta_p_plot,
        velocity_3d_plot=velocity_3d_plot,
        heat_map_mu_L=heat_map_mu_L,
        heat_map_kappa_L=heat_map_kappa_L,
        heat_map_deltaP_L=heat_map_deltaP_L,
        heat_map_mu_kappa=heat_map_mu_kappa,
        heat_map_mu_deltaP=heat_map_mu_deltaP,
        heat_map_kappa_deltaP=heat_map_kappa_deltaP,
    )

if __name__ == '__main__':
    app.run(debug=True)
