import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import Parameters14 as par
from time import time

## Defining PDE -------------------------------------------------------------------

pi = np.pi

# Initial Condition
def IC(x):
    return tf.sin(pi * x)

#Boundary Conditions
def BC(t,x):
    n = x.shape[0]
    return tf.zeros((n,1))

#PDE Residual
def func_residual(t, x, u_t, u_xx):
    return u_t - u_xx + tf.exp(-t)*(tf.sin(pi*x) - pi**2*tf.sin(pi*x))

if par.N_e:
    def exact(t, x):
        return tf.exp(-t)*tf.sin(pi*x)


# Points of Analysis ------------------------------------------------------------------- 

# Setting the domain
lb = tf.constant([par.tmin, par.xmin])
ub = tf.constant([par.tmax, par.xmax])

tf.random.set_seed(10)

# Assuming uniform sampling for IC (Usually Latin Hyprcube Sampling done)
t_0 = tf.ones((par.N_0,1)) * par.tmin
x_0 = tf.random.uniform((par.N_0,1), par.xmin, par.xmax)
X_0 = tf.concat([t_0, x_0], axis=1)
u_0 = IC(x_0)

# Assuming uniform sampling for BC (Usually Latin Hyprcube Sampling done)
if par.BC != 'none':
    t_b = tf.random.uniform((par.N_b,1), par.tmin, par.tmax)
    if par.BC=='both':
        x_b = par.xmin + (par.xmax - par.xmin) * tf.keras.backend.random_bernoulli((par.N_b,1), 0.5)
    if par.BC=='x=1':
        x_b = tf.ones((par.N_b,1))*par.xmax
    if par.BC=='x=-1':
        x_b = tf.ones((par.N_b,1))*par.xmin   
    X_b = tf.concat([t_b, x_b], axis=1)
    u_b = BC(t_b, x_0)

# Assuming Uniform Sampling inside the domain
t_r = tf.random.uniform((par.N_r,1), par.tmin, par.tmax)
x_r = tf.random.uniform((par.N_r,1), par.xmin, par.xmax)
X_r = tf.concat([t_r, x_r], axis=1)

# Internal Points Sampling
choice = list(np.random.choice(t_r.shape[0], size=par.N_e))
if par.N_e:
    if par.t_const > par.tmin and par.t_const < par.tmax:
        t_e = tf.constant(par.t_const, shape=(par.N_e,1), dtype=tf.dtypes.float32)
    else:
        t_e = tf.gather(t_r, indices=choice)
    
    if par.x_const > par.xmin and par.x_const < par.xmax:
        x_e = tf.constant(par.x_const, shape=(par.N_e,1), dtype=tf.dtypes.float32)
    else:
        x_e = tf.gather(x_r, indices=choice)

    X_e = tf.concat([t_e, x_e], axis=1)
    u_e = exact(t_e, x_e)

# Collect boundary and inital data in lists
if par.BC == 'none':
    if par.N_e:
        X_data = [X_0, X_e]
        u_data = [u_0, u_e]
    else:
        X_data = [X_0]
        u_data = [u_0]
else:
    if par.N_e:    
        X_data = [X_0, X_b, X_e]
        u_data = [u_0, u_b, u_e]
    else:
        X_data = [X_0, X_b]
        u_data = [u_0, u_b]

# Plotting the location of analysis points -------------------------------------------------------------------
if par.plot_points:
    figure=plt.figure(figsize=(16,9))
    plt.scatter(t_0, x_0, c='r', marker='x')
    if par.BC != 'none':
        plt.scatter(t_b, x_b, c='g', marker='x')
    plt.scatter(t_r, x_r, c='b', marker='.', s=1)
    if par.N_e:
        plt.scatter(t_e, x_e, c='black', marker='o', s=10)
    plt.xlabel('t')
    plt.ylabel('x')
    if par.plot_points_savefig:
        plt.savefig(par.plot_points_savefig)
    plt.show()
    
    if par.plot_points_savedat:
        t_0_write = t_0.numpy()
        t_r_write = t_r.numpy()
        if par.BC=='none':
            if par.N_e:
                t_e_write = t_e.numpy()
                t_points = np.vstack((t_0_write, t_r_write, t_e_write))
            else:
                t_points = np.vstack((t_0_write, t_r_write))
        else:
            t_b_write = t_b.numpy()
            if par.N_e:
                t_e_write = t_e.numpy()
                t_points = np.vstack((t_0_write, t_b_write, t_r_write, t_e_write))
            else:
                t_points = np.vstack((t_0_write, t_b_write, t_r_write))

        x_0_write = x_0.numpy()
        x_r_write = x_r.numpy()
        if par.BC == 'none':
            if par.N_e:
                x_e_write = x_e.numpy()
                x_points = np.vstack((x_0_write, x_r_write, x_e_write))
            else:
                x_points = np.vstack((x_0_write, x_r_write))
        else:
            x_b_write = x_b.numpy()
            if par.N_e:
                x_e_write = x_e.numpy()
                x_points = np.vstack((x_0_write, x_b_write, x_r_write, x_e_write))
            else:
                x_points = np.vstack((x_0_write, x_b_write, x_r_write))
        
        write_points = np.hstack((t_points, x_points))
        np.savetxt(par.plot_points_savedat, write_points)

# Network Architecture -------------------------------------------------------------------

def init_model(num_hidden_layers=par.num_hidden_layers, num_neurons_per_layer=par.num_neurons_per_layer):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(2,)) # Input is two-dimensional (time + one spatial dimension)

    scaling_layer = tf.keras.layers.Lambda(lambda x: 2.0*(x - lb)/(ub - lb) - 1.0) 
    model.add(scaling_layer) # Introduce a scaling layer to map input to [lb, ub]

    for _ in range(num_hidden_layers): # Append hidden layers
        model.add(tf.keras.layers.Dense(num_neurons_per_layer, activation=tf.keras.activations.get('tanh')))
    
    model.add(tf.keras.layers.Dense(1)) # Output is one-dimensional (Value of u)
    
    return model


# Functions for Loss and Gradients -------------------------------------------------------------------

# Function for Residual
def get_r(model, X_r):

    with tf.GradientTape(persistent=True) as tape:
        # Split t and x to compute partial derivatives
        t, x = X_r[:, 0:1], X_r[:,1:2]

        # Variables t and x are watched during tape to compute derivatives u_t and u_x
        tape.watch(t)
        tape.watch(x)

        # Determine residual # Forward Propogation
        u = model(tf.stack([t[:,0], x[:,0]], axis=1))
    
        # Compute gradient u_x within the GradientTape since we need second derivatives
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)       
    
    u_xx = tape.gradient(u_x, x)

    del tape

    return func_residual(t, x, u_t, u_xx)


# Function for Loss
def compute_loss(model, X_r, X_data, u_data):
    
    # Compute Collocation Point Loss
    r = get_r(model, X_r)
    phi_r = tf.reduce_mean(tf.square(r))
    
    # Initialize loss
    loss = phi_r
    
    # Add IC Loss and BC Loss to the loss
    for i in range(len(X_data)):
        u_pred = model(X_data[i])
        loss += tf.reduce_mean(tf.square(u_data[i] - u_pred))
    
    if par.N_e:
        u_pred_exact = model(X_data[-1])
        loss += tf.reduce_mean(tf.square(u_data[-1] - u_pred_exact))

    return loss


# Gradient of Loss Function
def get_grad(model, X_r, X_data, u_data):
    
    with tf.GradientTape(persistent=True) as tape:
        # This tape is for derivatives with
        # respect to trainable variables
        tape.watch(model.trainable_variables)
        loss = compute_loss(model, X_r, X_data, u_data)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g


# Training -------------------------------------------------------------------

model = init_model()
lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([1000,3000],[1e-2,1e-3,5e-4])
optim = tf.keras.optimizers.Adam(learning_rate=lr)

# Define one training step as a TensorFlow function to increase speed of training
@tf.function
def train_step():
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad(model, X_r, X_data, u_data)
    
    # Perform gradient descent step
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))
    
    return loss

hist = []

t0 = time()

for i in range(par.epochs+1):
    
    loss = train_step()
    
    # Append current loss to hist
    hist.append(loss.numpy())
    
    # Output current loss after 50 iterates
    if i%50 == 0:
        print('It {:05d}: loss = {:10.8e}'.format(i,loss))
        
# Print computation time
print('\nComputation time: {} seconds'.format(time()-t0))


# Prediction -------------------------------------------------------------------
tspace = np.linspace(lb[0], ub[0], par.mesh)
xspace = np.linspace(lb[1], ub[1], par.mesh)
T, X = np.meshgrid(tspace, xspace)
Xgrid = np.vstack([T.flatten(),X.flatten()]).T

# Determine predictions of u(t, x)
upred = model(Xgrid)
U = upred.numpy().reshape(par.mesh,par.mesh)

# Exact Solution
a = np.exp(-T)
b =  np.sin(pi*X)
sol = np.multiply(a ,b)

# Plotting difference between predicted and exact solution 
if par.plot_residual_solution:
    fig = plt.figure(figsize=(16,9))
    plt.scatter(T,X,s=1,c=sol-U, cmap='seismic')
    plt.colorbar()
    plt.xlabel('t')
    plt.ylabel('x')
    climlow = np.abs(np.min(sol-U))
    climmax = np.abs(np.max(sol-U))
    lim = max(climlow, climmax)
    plt.clim(-lim, lim)
    if par.plot_residual_solution_savefig:
        plt.savefig(par.plot_residual_solution_savefig)
    plt.show()

# Save the array for predicted solution
if par.save_data:
    T_write = T.reshape((T.shape[0]*T.shape[1], 1))
    X_write = X.reshape((X.shape[0]*X.shape[1], 1))
    U_write = U.reshape((U.shape[0]*U.shape[1], 1))
    sol_write = sol.reshape((sol.shape[0]*sol.shape[1], 1))
    write_array = np.hstack((T_write, X_write, U_write, sol_write))
    np.savetxt(par.save_data, write_array)

# Plotting the loss vs epoch
if par.plot_loss:
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.semilogy(range(len(hist)), hist,'k-')
    ax.set_xlabel('$n_{epoch}$')
    ax.set_ylabel('$\\phi_{n_{epoch}}$')
    if par.plot_loss_savefig:
        plt.savefig(par.plot_loss_savefig)
    plt.show()




