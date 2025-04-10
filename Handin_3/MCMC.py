import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def target_function(vec):
    x, y = vec
    return x**2 + y**2 + 4 * np.sin(x) + 4 * np.cos(y) + 0.5*x*y


# Metropolis-Hastings MCMC algorithm for 2D function
def metropolis_hastings_minimization(target_function, initial_point, num_iterations, step_size=0.1, temperature=1.0):
    # Initialize the chain with the initial point
    current_point = np.array(initial_point)
    current_value = target_function(current_point)

    Ndims = len(initial_point)
    
    # List to store the sequence of accepted points
    accepted_points = [current_point]
    
    for i in range(num_iterations):
        # Generate a new candidate by perturbing the current point (both x and y)
        proposal = current_point + np.random.normal(0, step_size, size=Ndims)
        
        # Evaluate the target function at the proposed point
        proposal_value = target_function(proposal)
        
        # Compute the acceptance probability
        if proposal_value < current_value:  # Always accept if the new value is better
            accept = True
        else:
            # Accept with a probability based on the temperature and function difference
            acceptance_probability = np.exp(-(proposal_value - current_value) / temperature)
            accept = np.random.rand() < acceptance_probability
        
        # If accepted, update the current point
        if accept:
            current_point = proposal
            current_value = proposal_value
            accepted_points.append(current_point)
    
    return np.array(accepted_points)

# Run the MCMC algorithm for 2D
initial_point = [5.0, 5.0]  # Initial guess for the minimization
num_iterations = 10_000
step_size = 0.1  # Step size for proposal distribution
temperature = 1.0  # Temperature for acceptance probability

# Run the MCMC to find the minimum
accepted_points = metropolis_hastings_minimization(target_function, initial_point, num_iterations, step_size, temperature)

# Extract the best point (lowest function value)
best_point = accepted_points[np.argmin([target_function(p) for p in accepted_points])]
print("Best point found:", best_point)

# Create a meshgrid for the 2D function surface (for heatmap/contour)
x_vals = np.linspace(-10, 10, 400)
y_vals = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = target_function([X, Y])

# Create subplots (2 rows, 1 column)
fig = plt.figure(figsize=(14, 8))

# Subplot 1: Heatmap or Contour Plot
ax1 = fig.add_subplot(121)  # 1st subplot (left side)
# Choose between heatmap or contour plot:
ax1.imshow(Z, extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()], origin='lower', cmap='viridis', alpha=0.7)
ax1.set_title("Heatmap of f(x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.scatter(accepted_points[:, 0], accepted_points[:, 1], color="r", s=10, label="MCMC Path")
ax1.scatter(best_point[0], best_point[1], color="g", label="Best Point Found", s=50)
ax1.legend()

# Subplot 2: 3D Surface Plot
ax2 = fig.add_subplot(122, projection='3d')  # 2nd subplot (right side)
# Plot the surface of the target function
ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7)
ax2.set_title("3D Surface Plot of f(x, y)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x, y)")

# Plot the MCMC path on the 3D surface
ax2.scatter(accepted_points[:, 0], accepted_points[:, 1], 
            [target_function(p) for p in accepted_points], 
            color="r", s=10, label="MCMC Path")
ax2.scatter(best_point[0], best_point[1], target_function(best_point), 
            color="g", label="Best Point Found", s=50)

# Show the plot
plt.tight_layout()
fig.savefig("figures/tests/MC.png", bbox_inches="tight", dpi=600)
