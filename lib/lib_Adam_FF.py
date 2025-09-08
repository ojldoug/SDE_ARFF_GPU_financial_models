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
        # TF model: no internal normalization or z_std/z_mean rescaling
        return Functions.beta(params, x)

    def diffusion_cov(params, x, diff_type):
        cov_vectors = Functions.beta(params, x)
        
        if diff_type == "diagonal":
            # softplus + EPS, same as TF
            return vmap(jnp.diag)(jax.nn.softplus(cov_vectors) + EPS)

    def diffusion(params, x, diff_type):
        if diff_type == "diagonal":
            return jnp.sqrt(Functions.diffusion_cov(params, x, diff_type))
    
        diff_vectors = Functions.beta(params, x)

        # arrange vectors into LT matrix
        T = diff_vectors.shape[-1]
        D = int((math.sqrt(1 + 8 * T) - 1) / 2)
        r, c = jnp.tril_indices(D)
        LT = jnp.zeros((x.shape[0], D, D)).at[..., r, c].set(diff_vectors)

        # apply safety on diagonal
        diag_idx = jnp.arange(D)
        LT = LT.at[..., diag_idx, diag_idx].set(jax.nn.softplus(jnp.diagonal(LT, axis1=-2, axis2=-1)) + EPS)
    
        if diff_type == "triangular":
            return LT 

        # make symmetric 
        return jnp.matmul(LT, jnp.swapaxes(LT, -1, -2))


def multivariate_nll_from_scale_tril(scale_tril, delta):
    # solve L y = delta for y
    # jax.scipy.linalg.solve_triangular does not always broadcast over batch, so vmap safely
    def solve_one(Lb, db):
        # lower=True, trans='N'
        return jax.scipy.linalg.solve_triangular(Lb, db, lower=True)

    y = vmap(solve_one)(scale_tril, delta)        
    quad = jnp.sum(y*y, axis=1)                
    diag = jnp.diagonal(scale_tril, axis1=-2, axis2=-1)  
    logdet = 2.0 * jnp.sum(jnp.log(diag), axis=1)       
    return quad, logdet

    
class AdamTrain:
    
    def nll_loss(drift_param, diff_param, x0, x1, h, diff_type):
        D = x1.shape[1]
        f = Functions.drift(drift_param, x0)
        delta = x1 - x0 - h * f   # shape (B, D)

        if diff_type == "diagonal":
            # TF: network outputs std, scale = sqrt(h) * std
            Sigma = jax.nn.softplus(Functions.beta(diff_param, x0)) + EPS
            quad = jnp.sum((delta ** 2) / (Sigma * h), axis=1)
            logdet = jnp.sum(jnp.log(Sigma * h), axis=1)
        else:
            scale_tril = Functions.diffusion(diff_param, x0, diff_type) * jnp.sqrt(h)
            if diff_type == "symmetric":
                Sigma = jnp.matmul(scale_tril, jnp.swapaxes(scale_tril, -1, -2))
                scale_tril = jnp.linalg.cholesky(Sigma)
            quad, logdet = multivariate_nll_from_scale_tril(scale_tril, delta)
            
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


