# Domain Limits
tmin = 0.0
tmax = 2.0
xmin = -1.0
xmax = 1.0

# Number of Points
N_0 = 100 # Initial Condition
N_b = 100 # Boundary Condition
N_r = 2000 # Collocation Points
N_e = 0 # Internal Points

# Specifying Boundary Conditions
BC = 'both'  # Can take values ['both', 'x=1', x=-1', 'none']

# Only to adjust distribution of internal points.
# Specify point outside domain if not needed.
# 1 of these MUST be outside limits.
x_const =-2
t_const = 1.5

# Architecture Specifications
num_hidden_layers = 8
num_neurons_per_layer = 20
epochs = 5000

# For analysing final prediction, and additional features
mesh = 1000 

# Additional Features
plot_points = True
plot_points_savefig = False # False, or give the name to save the figure
plot_points_savedat = False # False, or give the name to save the .dat file
plot_residual_solution = True
plot_residual_solution_savefig = False # False, or give the name to save the figure
plot_loss = False
plot_loss_savefig = False # False, or give the name to save the figure
save_data = False # False, or give the name to save the .dat file