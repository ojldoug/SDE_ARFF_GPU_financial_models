import jax
import jax.numpy as jnp
from jax import random, lax

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def plot_loss(times, losses):
    plt.plot(times, losses)
    plt.xlabel('time (s)')
    plt.ylabel('loss')


class PlotResults:
    def __init__(self, Functions):
        self.Functions = Functions
        
    def plot_train_v_true(self, param, x, diff_type=None, true_func=None, z=None):
        if z is not None:
            x = x[:len(z)]
        plot_funcs = {1: self.plot_1D, 2: self.plot_2D}
        plot_func = plot_funcs.get(x.shape[1], self.plot_ND)
        plot_func(true_func, param, x, diff_type, z)
    
    def plot_1D(self, true_func, param, x, diff_type, z):
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        
        if z is not None:
            ax.scatter(x, z, color="black")
        
        x = jnp.sort(x, axis=0)
    
        if true_func is not None:
            ax.plot(x, true_func(x), color="green")
        
        if diff_type is None:
            ax.plot(x, self.Functions.drift(param, x))
        else:
            ax.plot(x, self.Functions.diffusion(param, x, diff_type).reshape(-1, 1))
        plt.show()
    
    def plot_2D(self, true_func, param, x, diff_type, z):
        yD = param["amp"].shape[1]
        col = 4 if z is not None else 2
    
        true = true_func(x)
        if diff_type is None:
            trained = self.Functions.drift(param, x)
        else:
            trained = self.Functions.diffusion(param, x, diff_type)
            if diff_type == "diagonal":
                true, trained = (jnp.diagonal(arr, axis1=1, axis2=2) for arr in (true, trained))
            else:
                LT_idx = jnp.tril_indices(yD)
                true, trained = (arr[:, LT_idx[0], LT_idx[1]] for arr in (true, trained))
    
        if z is not None:
            intermediate = self.Functions.beta(param, x)
        
        fig, ax = plt.subplots(yD, col, figsize=(5*col, 4*yD), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})
    
        def plotting_2D(d, data1, data2, title1, title2, col=0):
            norms = Normalize(vmin=min(data1[:, d].min(), data2[:, d].min()), 
                              vmax=max(data1[:, d].max(), data2[:, d].max())) 
    
            plot_1 = ax[d, col].scatter(x[:, 0], x[:, 1], c=data1[:, d], cmap='viridis', s=20, norm=norms)
            ax[d, col+1].scatter(x[:, 0], x[:, 1], c=data2[:, d], cmap='viridis', s=20, norm=norms)
    
            fig.colorbar(plot_1, ax=ax[d, col-2], orientation='vertical', fraction=0.02, pad=0.04)
            if d == 0:
                ax[0, col].set_title(title1, fontsize=12)
                ax[0, col+1].set_title(title2, fontsize=12)
            
        for d in range(yD):
            plotting_2D(d, trained, true, "Trained", "True", col-2)
            if z is not None:
                plotting_2D(d, z, intermediate, "Training Data", "Intermediate")
    
        plt.show()

    
    def plot_ND(self, true_func, param, x, diff_type, z):
        yD = param["amp"].shape[1]
        col = 2 if z is not None else 1
        
        true = true_func(x)
        if diff_type is None:
            title = 'f'
            trained = self.Functions.drift(param, x)
        else:
            title = r'\sigma'
            trained = self.Functions.diffusion(param, x, diff_type)
            if diff_type == "diagonal":
                true, trained = (jnp.diagonal(arr, axis1=1, axis2=2) for arr in (true, trained))
            else:
                LT_idx = jnp.tril_indices(yD)
                true, trained = (arr[:, LT_idx[0], LT_idx[1]] for arr in (true, trained))
    
        if z is not None:
            intermediate = self.Functions.beta(param, x)
            
        fig, ax = plt.subplots(yD, col, figsize=(5*col, 4*yD), gridspec_kw={'hspace': 0.12, 'wspace': 0.19})
        if ax.ndim == 1:
            ax = ax[:, None]
        
        def plotting_ND(d, data1, data2, col=0):
            ax[d, col].scatter(data1[:, d], data2[:, d], s=5, alpha=0.5) 
            
            min_val = min(data1[:, d].min(), data2[:, d].min())
            max_val = max(data1[:, d].max(), data2[:, d].max())
            ax[d, col].plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
    
            ax[d, col].set_xlabel("True")
            ax[d, col].set_ylabel("Trained")
            #ax[d, col].set_title(rf'${title}_{{{d}}}$')
            ax[d, col].set_aspect('equal', adjustable='box')
            
        for d in range(yD):
            plotting_ND(d, true, trained, col-1)
            if z is not None:
                plotting_ND(d, z, intermediate)
            
        plt.show()
    
    def plot_trajectories(self, drift_param, diff_param, x_domain, xlim,
                          n_trajectories, trajectory_time, h, diff_type, D):
    
        def simulate_euler_maruyama(key, drift_fn, diffusion_fn, x0, h, grid_resolution, xlim):
            """
            Parallel Euler-Maruyama simulation with trajectory termination.
            - xlim: (D, 2) array with min/max per dimension
            """
            n_trajectories = x0.shape[0]
            T = grid_resolution
    
            # Generate all Wiener increments in parallel
            key, subkey = random.split(key)
            dW = random.normal(subkey, shape=(T-1, n_trajectories, D)) * jnp.sqrt(h)
    
            # Initialize state: (x_t, alive_mask)
            alive_init = jnp.ones((n_trajectories,), dtype=bool)
            init_state = (x0, alive_init)
    
            def euler_step(carry, dW_t):
                x_t, alive = carry
                drift = drift_fn(x_t)
                diffusion = diffusion_fn(x_t).reshape(n_trajectories, D, D)
                diffusion_term = jnp.einsum('nij,nj->ni', diffusion, dW_t)
                x_next = x_t + drift * h + diffusion_term
                print(x_t.shape, x_next.shape)
    
                # Update alive mask
                within_limits = jnp.all((x_next >= xlim[:, 0]) & (x_next <= xlim[:, 1]), axis=1)
                alive = alive & within_limits
    
                # Fill terminated trajectories with NaN
                x_next = jnp.where(alive[:, None], x_next, jnp.nan)
                return (x_next, alive), x_next
    
            # Run lax.scan over all time steps
            (_, _), xs = lax.scan(euler_step, init_state, dW)
    
            # Concatenate initial condition
            x_full = jnp.concatenate([x0[:, :, None], xs.transpose(1, 2, 0)], axis=-1)  # (N, D, T)
            return x_full
    
        # Prepare parameters
        grid_resolution = int(trajectory_time / h) + 1
        key = random.PRNGKey(0)
        x0 = random.uniform(key, shape=(n_trajectories, x_domain.shape[0]),
                            minval=x_domain[:, 0], maxval=x_domain[:, 1])
    
        drift_fn = lambda x: self.Functions.drift(drift_param, x)
        diffusion_fn = lambda x: self.Functions.diffusion(diff_param, x, diff_type)
    
        # Simulate all trajectories
        x = simulate_euler_maruyama(key, drift_fn, diffusion_fn, x0, h, grid_resolution, xlim)
        print(x.shape)
        # Time vector
        t = jnp.linspace(0, trajectory_time, grid_resolution)
    
        # Plot each dimension
        D = x.shape[1]
        for d in range(D):
            plt.figure(figsize=(10, 6))
            for n in range(n_trajectories):
                plt.plot(t, x[n, d, :], alpha=0.6)
            plt.title(f"Trajectories for Dimension {d}")
            plt.xlabel("Time")
            plt.ylabel(f"x[{d}]")
            plt.grid(True)
            plt.show()
    
        return x









    

    