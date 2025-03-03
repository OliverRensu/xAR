import torch
import numpy as np
import math

def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur

@torch.no_grad()
def euler_sampler(
        model,
        latents,
        y,
        scale_index,
        condition=None,
        num_steps=50,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        mask=None
        ):
    # setup conditioning
    _dtype = latents.dtype    
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device
    cfg = cfg_scale
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            #cfg_scale = (cfg-1.)*i/num_steps+1
            x_cur = x_next
            if condition is not None and i==0:
                model_input = torch.cat([condition, x_next], dim=1)
            else:
                model_input = x_next
            if t_cur <= guidance_high or i==0:
                model_input = torch.cat([model_input, model_input], dim=0)
                y_cur = torch.cat([y, torch.ones_like(y).cuda()*1000], dim=0)
            else:
                y_cur = y
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            if i==0:
                d_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                torch.cat([torch.zeros(model_input.size(0)*(1 if scale_index else 0)).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
                True, scale_index, mask).to(torch.float64)
            else:
                d_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                time_input.to(dtype=_dtype), 
                False, scale_index).to(torch.float64)
            if t_cur <= guidance_high or i ==0:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)  
                else:
                    d_cur = d_cur_cond    
            x_next = x_cur + (t_next - t_cur) * d_cur          
    return x_next

@torch.no_grad()
def euler_maruyama_sampler(
        model,
        latents,
        y,
        scale_index,
        condition=None,
        num_steps=50,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        mask=None
        ):
    # setup conditioning

    _dtype = latents.dtype
    
    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next

            if condition is not None and i==0:
                model_input = torch.cat([condition, x_next], dim=1)
            else:
                model_input = x_next
            if t_cur <= guidance_high or i==0:
                model_input = torch.cat([model_input, model_input], dim=0)
                y_cur = torch.cat([y, torch.ones_like(y).cuda()*1000], dim=0)
            else:
                y_cur = y

            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)            
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # compute drift
            if i==0:
                v_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                torch.cat([torch.zeros(model_input.size(0)*(1 if scale_index else 0)).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
                True, scale_index, mask).to(torch.float64)
            else:
                v_cur = model(
                model_input.to(dtype=_dtype), y_cur, 
                time_input.to(dtype=_dtype), 
                False, scale_index).to(torch.float64)
            model_input=model_input[:, -64:]
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur

            if t_cur <= guidance_high or i ==0:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)  
                else:
                    d_cur = d_cur_cond    
            x_next =  x_cur + d_cur * dt + torch.sqrt(diffusion) * deps
    
    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    
    if condition is not None and i==0:
        model_input = torch.cat([condition, x_next], dim=1)
    else:
        model_input = x_next
    if t_cur <= guidance_high or i==0:
        model_input = torch.cat([model_input, model_input], dim=0)
        y_cur = torch.cat([y, torch.ones_like(y).cuda()*1000], dim=0)
    else:
        y_cur = y


    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur
    
    # compute drift
    if i==0:
        v_cur = model(
        model_input.to(dtype=_dtype), y_cur, 
        torch.cat([torch.zeros(model_input.size(0)*(1 if scale_index else 0)).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
        True, scale_index).to(torch.float64)
    else:
        v_cur = model(
        model_input.to(dtype=_dtype), y_cur, 
        time_input.to(dtype=_dtype), 
        False, scale_index).to(torch.float64)
    # v_cur = model(
    #             model_input.to(dtype=_dtype), y_cur, 
    #             torch.cat([torch.zeros(model_input.size(0)*scale_index).to(device=device, dtype=torch.float64), time_input.to(dtype=_dtype)], dim=0), 
    #             ).to(torch.float64)
    model_input=model_input[:, -64:]
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur

    if t_cur <= guidance_high or i ==0:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
            d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)  
        else:
            d_cur = d_cur_cond    
    # if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
    #     d_cur_cond, d_cur_uncond = d_cur.chunk(2)
    #     d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur
                    
    return mean_x
