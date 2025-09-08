import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap
from jax import random, grad, value_and_grad
from jax import nn
import jax.scipy.linalg
from functools import partial
jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

import time
import math
import numpy as np

EPS = 1e-13

class Functions:
    @jit
    def S(omega, x):
        x_proj = x @ omega
        return jnp.concatenate([jnp.cos(x_proj), jnp.sin(x_proj)], axis=-1)
        #return jnp.exp(1j * x_proj)
    
    @jit
    def beta(params, x):
        return (Functions.S(params["omega"], x) @ params["amp"])
    
    def drift(params, x):
        return Functions.beta(params, x)

    def diffusion_cov(params, x, diff_type="diagonal"):
        def softplus_minimal(x, beta=10.0, eps=1e-13):
            return jax.nn.softplus(beta * x) / beta + eps
            #return jnp.abs(x) + eps
            
        cov_vectors = Functions.beta(params, x)
        
        if diff_type == "diagonal":
            return vmap(jnp.diag)(softplus_minimal(cov_vectors))
            #return vmap(jnp.diag)(cov_vectors)
        
        T = params["amp"].shape[1]
        D = int((math.sqrt(1 + 8 * T) - 1) / 2)
        
        lt_i, lt_j = np.tril_indices(D)
        LT_idx = (jnp.array(lt_i), jnp.array(lt_j))
        
        def make_symmetric_matrix(vec):
            mat = jnp.zeros((D, D))
            mat = mat.at[LT_idx].set(vec)
            sym = mat + mat.T - jnp.diag(jnp.diag(mat))

            # # Apply softplus + eps to the diagonal *after* symmetrization
            # diag = softplus_minimal(jnp.diag(mat))
            # sym = sym.at[jnp.diag_indices(D)].set(diag)
            vals, vecs = jnp.linalg.eigh(sym)
            pos_vals = softplus_minimal(vals)
            spd = (vecs * pos_vals) @ vecs.T
            return spd
    
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

    
class AdamTrain:
    
    def nll_loss(drift_param, diff_param, x0, x1, h, diff_type):
        D = x1.shape[1]
        f = Functions.drift(drift_param, x0)
        delta = x1 - x0 - h * f   # shape (B, D)
        Sigma = Functions.diffusion_cov(diff_param, x0, diff_type) * h 

        if diff_type == "diagonal":
            # TF: network outputs std, scale = sqrt(h) * std
            Sigma_diag = jnp.diagonal(Sigma, axis1=1, axis2=2)
            quad = jnp.sum((delta ** 2) / (Sigma_diag), axis=1)
            logdet = jnp.sum(jnp.log(Sigma_diag), axis=1)
        else:
            def get_quad_and_logdet(S, d):
                sol, _ = jax.scipy.sparse.linalg.cg(lambda v: S @ v, d, tol=1e-6, maxiter=1e4)
                quad = jnp.dot(d, sol)
                _, ld = jnp.linalg.slogdet(S)
                return quad, ld
            quad, logdet = jax.vmap(get_quad_and_logdet)(Sigma, delta)
            
        losses = 0.5 * quad + 0.5 * logdet + 0.5 * D * jnp.log(2.0 * jnp.pi)
        return jnp.mean(losses)
    
    @partial(jit, static_argnames=['opt', 'diff_type'])
    def train_step(drift_param, diff_param, opt, opt_state_drift, opt_state_diff, x0, x1, h, diff_type):
        loss, grads = value_and_grad(AdamTrain.nll_loss, argnums=(0, 1))(
            drift_param, diff_param, x0, x1, h, diff_type
        )
        grads_drift, grads_diff = grads
        updates_drift, opt_state_drift = opt.update(grads_drift, opt_state_drift)
        updates_diff, opt_state_diff = opt.update(grads_diff, opt_state_diff)
        drift_param = optax.apply_updates(drift_param, updates_drift)
        diff_param = optax.apply_updates(diff_param, updates_diff)
        
        def global_norm(grads):
            return jnp.sqrt(sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(grads)))
        grad = global_norm(grads)
        return drift_param, diff_param, opt_state_drift, opt_state_diff, loss, grad
    
    def training_loop(hyperparam, drift_param, diff_param, x0, x1, h, diff_type, val_split):
        opt = optax.adam(hyperparam["learning_rate"], b1=0.9, b2=0.999, eps=1e-7)
        opt_state_drift = opt.init(drift_param)
        opt_state_diff = opt.init(diff_param)

        # TF: validation_split = last fraction of data
        n_val = int(x0.shape[0] * val_split)
        x0_train, x1_train = x0[:-n_val], x1[:-n_val]
        x0_val, x1_val = x0[-n_val:], x1[-n_val:]
        N_train = x0_train.shape[0]
        batch_size = N_train if hyperparam["batch_size"] is None else hyperparam["batch_size"]

        losses, val_losses, grads, times = [], [], [], []
        start_time = time.time()
        
        for epoch in range(hyperparam["epochs"]):
            # shuffle training data each epoch (only training, not val)
            perm = jax.random.permutation(jax.random.PRNGKey(epoch), N_train)
            x0_shuffled, x1_shuffled = x0_train[perm], x1_train[perm]

            for i in range(0, N_train, batch_size):
                x0_batch = x0_shuffled[i:i+batch_size]
                x1_batch = x1_shuffled[i:i+batch_size]
                drift_param, diff_param, opt_state_drift, opt_state_diff, loss, grad = AdamTrain.train_step(
                    drift_param, diff_param, opt, opt_state_drift, opt_state_diff,
                    x0_batch, x1_batch, h, diff_type
                )

            val_loss = AdamTrain.nll_loss(drift_param, diff_param, x0_val, x1_val, h, diff_type)
            
            losses.append(float(loss))
            val_losses.append(float(val_loss))
            times.append(time.time() - start_time)
            #grads.append(float(grad))
            print(f"\repoch {epoch}: loss = {loss:.4f}, val_loss = {val_loss:.4f}", end='')
        print() 

        return drift_param, diff_param, times, losses, val_losses#, grads


