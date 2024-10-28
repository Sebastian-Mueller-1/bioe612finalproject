from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

# init a flask app instance
app = Flask(__name__)

# Parameters
R_max = 10.0  # max r
r_min = 1e-6  # Min r to avoid singularity
Grid_L = R_max  # grid range for plotting
N = 400  # Grid resolution
V_max = 1e7  # Initial fixed maximum velocity for color scale

# make meshgrid
x = np.linspace(-Grid_L, Grid_L, N) # x axis from negative grid extent to max grid extent
y = np.linspace(-Grid_L, Grid_L, N) # same for y axis
X, Y = np.meshgrid(x, y)

def compute_velocity(mu, kappa, P0, Pe):
    '''This function computes the velocity using Darcy law parameters'''

    # Ensure mu and kappa do not fall below a minimum positive value
    mu = max(mu, 0.1)
    kappa = max(kappa, 0.1)

    delta_P = Pe - P0 # calc the pressure differential

    ln_term = np.log(R_max / r_min) # produce the ln term for radial integration

    # protect from divsion by zero error
    if ln_term == 0:
        ln_term = 1e-6

    # calculate Q (Darcy volumentric flow rate)
    Q = -(2 * np.pi * kappa * delta_P) / (mu * ln_term)

    # compute radial distance from orgion for each discrete point in grid (2d array)
    r = np.sqrt(X**2 + Y**2)

    # clip to prevent division by zero and bound for grpahing
    r_clipped = np.clip(r, r_min, R_max)

    # calculate the radial velocity
    v_r = Q / (2 * np.pi * r_clipped)

    # convert back to Cartisian coordinates
    Vx = v_r * (X / r_clipped)
    Vy = v_r * (Y / r_clipped)

    # gives velocity at each point irrespective of direction
    V = np.sqrt(Vx**2 + Vy**2)

    # Mask values outside the circle for better looking plot
    V[r > Grid_L] = np.nan

    return np.clip(V, 0, V_max)  # Return array with velocity values for plotting

# build the webapp route for rendering
@app.route("/", methods=["GET", "POST"])
def index():

    # Default parameters
    mu, kappa, P0, Pe = 2.0, 2.0, 100, 15.0
    if request.method == "POST":
        mu = float(request.form.get("mu", mu))
        kappa = float(request.form.get("kappa", kappa))
        P0 = float(request.form.get("P0", P0))
        Pe = float(request.form.get("Pe", Pe))

    # Compute velocity field if sliders are changes and new rquest called
    V = compute_velocity(mu, kappa, P0, Pe)
    dynamic_vmax = np.nanmax(V)  # Calculate max value for dynamic color scale scaling

    # Set up plot with dynamic color scaling
    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap = ax.imshow(
        np.log10(V + 1e-6),  # Log transformation for visualization only
        cmap="coolwarm",
        extent=(-Grid_L, Grid_L, -Grid_L, Grid_L),
        vmin=0,
        vmax=np.log10(dynamic_vmax + 1e-6)  # Dynamic color scale max value
    )
    ax.set_title("Point Source Fluid Velocity Heatmap")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(heatmap, ax=ax, label="Log (Base 10) of Velocity Magnitude")

    # Convert plot to PNG image and encode it
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)

    # Render template with the plot image
    return render_template("index.html", image_base64=image_base64, mu=mu, kappa=kappa, P0=P0, Pe=Pe)
