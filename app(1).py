from flask import Flask, render_template
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Define base parameters for sensitivity analysis
mu_base = 0.01
kappa_base = 0.1
delta_p_base = 100

# Define radial velocity function without dependency on r
def radial_velocity(mu, kappa, delta_p):
    return (delta_p * kappa) / (2 * np.pi * mu)

# Generate sensitivity data for varying each parameter
num_points = 100
mu_values = np.linspace(0.005, 0.02, num_points)
kappa_values = np.linspace(0.05, 0.2, num_points)
delta_p_values = np.linspace(50, 150, num_points)

velocity_mu = [radial_velocity(mu, kappa_base, delta_p_base) for mu in mu_values]
velocity_kappa = [radial_velocity(mu_base, kappa, delta_p_base) for kappa in kappa_values]
velocity_delta_p = [radial_velocity(mu_base, kappa_base, delta_p) for delta_p in delta_p_values]

# Create plotly figures for each parameter sensitivity
def create_plot(parameter_values, velocities, parameter_name, parameter_label):
    trace = go.Scatter(x=parameter_values, y=velocities, mode='lines', name=parameter_name)
    layout = go.Layout(
        title=f'Sensitivity of Radial Velocity to {parameter_label}',
        xaxis=dict(title=parameter_label),
        yaxis=dict(title='Radial Velocity')
    )
    fig = go.Figure(data=[trace], layout=layout)
    return pio.to_html(fig, full_html=False)

@app.route('/')
def index():
    mu_plot = create_plot(mu_values, velocity_mu, 'Viscosity (μ)', 'Viscosity (μ)')
    kappa_plot = create_plot(kappa_values, velocity_kappa, 'Permeability (κ)', 'Permeability (κ)')
    delta_p_plot = create_plot(delta_p_values, velocity_delta_p, 'Pressure Difference (ΔP)', 'Pressure Difference (ΔP)')
    return render_template('index.html', mu_plot=mu_plot, kappa_plot=kappa_plot, delta_p_plot=delta_p_plot)

if __name__ == '__main__':
    app.run(debug=True)
