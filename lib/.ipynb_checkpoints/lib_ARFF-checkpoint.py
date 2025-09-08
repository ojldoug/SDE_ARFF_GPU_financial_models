import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap
from jax import random, grad, value_and_grad
from jax import nn
from functools import partial
from matplotlib import pyplot as plt

import time
import math
import numpy as np

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

EPS = 1e-13

def split_data(key, val_split, *inputs):
    key, subkey = random.split(key)

    # obtain split indices
    N = inputs[0].shape[0]
    val_sample_size = int(N * val_split)
    permuted_idx = random.permutation(subkey, N)
    val_idx = permuted_idx[:val_sample_size]
    mask = jnp.ones(N, dtype=bool).at[val_idx].set(False)

    # split the data
    inputs_train = tuple(data[mask] for data in inputs)
    inputs_valid = tuple(data[~mask] for data in inputs)

    return inputs_train, inputs_valid, key


class Functions:
    @jit
    def S(omega, x):
        x_proj = x @ omega
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)
        #return jnp.exp(1j * x_proj)
    
    @jit
    def beta(params, x):
        return (Functions.S(params["omega"], x) @ params["amp"]).real
    
    def drift(params, x):
        drift_ = Functions.beta(params, x)
        return drift_

    def diffusion_cov(params, x, diff_type):
        cov_vectors = Functions.beta(params, x)
        
        if diff_type == "diagonal":
            #return vmap(jnp.diag)(jax.nn.softplus(cov_vectors) + EPS)
            return vmap(jnp.diag)(jnp.abs(cov_vectors) + EPS)
        
        T = params["amp"].shape[1]
        D = int((math.sqrt(1 + 8 * T) - 1) / 2)
        
        lt_i, lt_j = np.tril_indices(D)
        LT_idx = (jnp.array(lt_i), jnp.array(lt_j))
        
        def make_symmetric_matrix(vec):
            mat = jnp.zeros((D, D))
            mat = mat.at[LT_idx].set(vec)
            sym = mat + mat.T - jnp.diag(jnp.diag(mat))
            return sym
    
        return vmap(make_symmetric_matrix)(cov_vectors)

    def diffusion(params, x, diff_type):        
        cov = Functions.diffusion_cov(params, x, diff_type)

        if diff_type == "diagonal":
            return jnp.sqrt(cov)
        elif diff_type == "triangular":
            return jax.vmap(jnp.linalg.cholesky)(cov)
        def matrix_sqrtm(mat):
            vals, vecs = jnp.linalg.eigh(mat)
            sqrt_vals = jnp.sqrt(jnp.clip(vals, a_min=0.0))
            return (vecs * sqrt_vals) @ vecs.T
        return jax.vmap(matrix_sqrtm, in_axes=0, out_axes=0)(cov)


def nll_loss(drift_param, diff_param, x0, x1, h, diff_type):
    D = x1.shape[1]
    f = Functions.drift(drift_param, x0)
    delta = x1 - x0 - h * f
    var = Functions.diffusion_cov(diff_param, x0, diff_type) * h 

    if diff_type == "diagonal":
        var_vec = jnp.diagonal(var, axis1=1, axis2=2)
        quad = jnp.sum((delta ** 2) / var_vec, axis=1)
        logdet = jnp.sum(jnp.log(var_vec), axis=1)
    else:
        def get_quad_and_logdet(S, d):
            sol, _ = jax.scipy.sparse.linalg.cg(lambda v: S @ v, d, tol=1e-6, maxiter=1e4)
            quad = jnp.dot(d, sol)
            _, ld = jnp.linalg.slogdet(S)
            return quad, ld
        quad, logdet = jax.vmap(get_quad_and_logdet)(var, delta)
        
    losses = 0.5 * quad + 0.5 * logdet + 0.5 * D * jnp.log(2.0 * jnp.pi)
    return jnp.mean(losses)


class ARFFHyperparameters:
    def __init__(self, K=2**6, M_min=0, M_max=100, lambda_reg=2e-3, gamma=1, delta=0.1, name=None):
        self.K = K
        self.M_min = M_min
        self.M_max = M_max
        self.lambda_reg = lambda_reg
        self.gamma = gamma
        self.delta = delta
        self.name = name

        
class ARFFTrain:
    def __init__(self, resampling=True, metropolis_test=True):
        self.resampling = resampling
        self.metropolis_test = metropolis_test
       
    @staticmethod
    def get_Sigma(drift_param, y0, y1, x, h, diff_type):
        f = y1 - (y0 + h * Functions.drift(drift_param, x))
        if diff_type == "diagonal":
            Sigma = f ** 2 / h
        else:
            Sigma = jnp.matmul(f[:, :, None], f[:, :, None].transpose(0, 2, 1)) / h
            LT_idx_i, LT_idx_j = jnp.tril_indices(f.shape[1])
            Sigma = Sigma[:, LT_idx_i, LT_idx_j]
        return Sigma
        
    @staticmethod
    @jit
    def get_amp(x, y, lambda_reg, omega):
        S_ = Functions.S(omega, x) 
        A = jnp.matmul(jnp.conj(jnp.transpose(S_)), S_) + x.shape[0] * lambda_reg * jnp.eye(S_.shape[1])
        b = jnp.matmul(jnp.conj(jnp.transpose(S_)), y)
        
        #amp = jnp.linalg.solve(A, b)
        amp, _ = jax.scipy.sparse.linalg.cg(A, b, tol=1e-6, maxiter=20000)
        return amp

    @staticmethod
    @partial(jit, static_argnames=['RESAMPLING', 'METROPOLIS_TEST'])
    def ARFF_one_step(key, omega, amp, x, y, delta, lambda_reg, gamma, RESAMPLING=True, METROPOLIS_TEST=True):

        amp_norm = jnp.linalg.norm(jnp.reshape(amp, (-1, omega.shape[1])), axis=0)

        if RESAMPLING:
            amp_pmf = amp_norm / jnp.sum(amp_norm)
            key, subkey = random.split(key)
            omega = omega[:, random.choice(subkey, amp_norm.shape[0], shape=(omega.shape[1],), p=amp_pmf)]

        if METROPOLIS_TEST:
            key, subkey = random.split(key)
            dw = random.normal(subkey, omega.shape)
            omega_prime = omega + delta * dw
            
            amp_prime_norm = jnp.linalg.norm(jnp.reshape(ARFFTrain.get_amp(x, y, lambda_reg, omega_prime), (-1, omega.shape[1])), axis=0)

            key, subkey = random.split(key)
            omega = jnp.where((amp_prime_norm / amp_norm) ** gamma >= random.uniform(subkey, omega.shape[1]), omega_prime, omega)
               
        else:
            key, subkey = random.split(key)
            dw = random.normal(subkey, omega.shape)
            omega = omega + delta * dw

        amp = ARFFTrain.get_amp(x, y, lambda_reg, omega)

        return omega, amp, key

    def ARFF_loop(self, key, hyperparam, x, y, val_split):
        start_time = time.time()
        (x, y), (x_valid, y_valid), key = split_data(key, val_split, x, y)

        omega = jnp.zeros((x.shape[1], hyperparam.K))
        amp = ARFFTrain.get_amp(x, y, hyperparam.lambda_reg, omega)

        val_errors = jnp.zeros(hyperparam.M_max)
        val_error_min = jnp.inf
        moving_sum = 0
        moving_avg = jnp.zeros(hyperparam.M_max)
        min_moving_avg = jnp.inf
        moving_avg_len = 5
        min_index = 0
        break_iterations = 5
        
        for i in range(hyperparam.M_max):
            omega, amp, key = ARFFTrain.ARFF_one_step(key, omega, amp, x, y, 
                                                      hyperparam.delta, hyperparam.lambda_reg,
                                                      hyperparam.gamma, 
                                                      RESAMPLING=self.resampling,
                                                      METROPOLIS_TEST=self.metropolis_test)
            
            val_error = jnp.mean(jnp.abs(Functions.beta({"omega": omega, "amp": amp}, x_valid) - y_valid) ** 2)
            val_errors = val_errors.at[i].set(val_error)

            # Update moving sum: add current error
            moving_sum += val_error
         
            # Subtract the error that's no longer in the window if window is full
            if i >= moving_avg_len:
                moving_sum -= val_errors[i - moving_avg_len]
   
            # Compute moving average
            window_size = i + 1 if i < moving_avg_len else moving_avg_len
            moving_avg = moving_avg.at[i].set(moving_sum / window_size)
  
            if moving_avg[i] < min_moving_avg:
                min_moving_avg = moving_avg[i]
                min_index = i
       
            if min_index + break_iterations < i and i > hyperparam.M_min:
                break
            
            if val_error < val_error_min:
                end_time = time.time()
                val_errors_min = val_error
                param = {"omega": omega, "amp": amp}

            if i % 1 == 0 or i == hyperparam.M_max - 1:
                print(f"\r{hyperparam.name} epoch: {i}", end='')
        print()
        
        return param, val_errors[:i], moving_avg[:i], end_time-start_time, key

    def train_model(self, key, drift_hyperparam, diff_hyperparam, y0, y1, h, x=None, YinX=True, val_split=0.1, ARFF_val_split=0.1, diff_type="diagonal", plot=True):
        if x is None:
            x = y0
        elif YinX:
            x = jnp.concatenate((y0, x), axis=1)

        (x, y0, y1), (x_valid, y0_valid, y1_valid), key = split_data(key, val_split, x, y0, y1)

        # calculate point-wise drift
        z_start = time.time()
        z = (y1 - y0)/h
        z_time = time.time() - z_start

        # train for z
        drift_param, drift_ve, drift_moving_avg, drift_time, key = ARFFTrain.ARFF_loop(self, key, drift_hyperparam, x, z, ARFF_val_split)
        plot and plot_loss(drift_ve, drift_moving_avg)
            
        # calculate point-wise diffusion
        Sigma_start = time.time()
        Sigma = ARFFTrain.get_Sigma(drift_param, y0, y1, x, h, diff_type)
        print(np.any(Sigma < 0))
        Sigma_time = time.time() - Sigma_start

        # train for global diffusion
        diff_param, diff_ve, diff_moving_avg, diff_time, key = ARFFTrain.ARFF_loop(self, key, diff_hyperparam, x, Sigma, ARFF_val_split)
        plot and plot_loss(diff_ve, diff_moving_avg)
        
        # outputs
        loss = nll_loss(drift_param, diff_param, y0, y1, h, diff_type)
        val_loss = nll_loss(drift_param, diff_param, y0_valid, y1_valid, h, diff_type)
        training_time = z_time + drift_time + Sigma_time + diff_time
        print(f"loss = {loss:.4f}, val_loss = {val_loss:.4f}, time = {training_time:.4f}s")
        
        return drift_param, diff_param, training_time, loss, val_loss, z, Sigma


def plot_loss(ve, moving_avg):
    plt.semilogy(ve, label="Validation Error")
    plt.semilogy(moving_avg, label="Moving Average")

    plt.title('ARFF Loss', fontsize=12)
    plt.xlabel(r'$M$', fontsize=12)
    plt.legend()
    plt.show()
    