import torch

import tinycudann as tcnn
import numpy as np
import nvdiffrast.torch as dr
from .utils import pixel_grid

def generate_envir_map_dir(envmap_h, envmap_w, is_jittor=False):
    lat_step_size = np.pi / envmap_h
    lng_step_size = 2 * np.pi / envmap_w
    phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, envmap_h), 
                                torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, envmap_w)], indexing='ij')

    sin_phi = torch.sin(torch.pi / 2 - phi)  # [envH, envW]
    light_area_weight = 4 * torch.pi * sin_phi / torch.sum(sin_phi)  # [envH, envW]
    assert 0 not in light_area_weight, "There shouldn't be light pixel that doesn't contribute"
    light_area_weight = light_area_weight.to(torch.float32).reshape(-1) # [envH * envW, ]
    if is_jittor:
        phi_jittor, theta_jittor = lat_step_size * (torch.rand_like(phi) - 0.5),  lng_step_size * (torch.rand_like(theta) - 0.5)
        phi, theta = phi + phi_jittor, theta + theta_jittor

    view_dirs = torch.stack([   torch.cos(theta) * torch.cos(phi), 
                                torch.sin(theta) * torch.cos(phi), 
                                torch.sin(phi)], dim=-1).view(-1, 3)    # [envH * envW, 3]
    
    return light_area_weight, view_dirs

class _MLP(torch.nn.Module):
    def __init__(self, cfg, loss_scale=1.0):
        super(_MLP, self).__init__()
        self.loss_scale = loss_scale
        net = (torch.nn.Linear(cfg['n_input_dims'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        for i in range(cfg['n_hidden_layers']-1):
            net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_neurons'], bias=False), torch.nn.ReLU())
        net = net + (torch.nn.Linear(cfg['n_neurons'], cfg['n_output_dims'], bias=False),)
        self.net = torch.nn.Sequential(*net).cuda()
        
        self.net.apply(self._init_weights)
        
        if self.loss_scale != 1.0:
            self.net.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * self.loss_scale, ))

    def forward(self, x):
        return self.net(x.to(torch.float32))

    @staticmethod
    def _init_weights(m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

class MLPTexture3D(torch.nn.Module):
    def __init__(self, AABB, channels = 3, internal_dims = 32, hidden = 2, min_max = None):
        super(MLPTexture3D, self).__init__()

        self.channels = channels
        self.internal_dims = internal_dims
        self.AABB = AABB.cuda()
        self.AABB = (self.AABB[0:3], self.AABB[3:6])
        self.min_max = min_max

        # Setup positional encoding, see https://github.com/NVlabs/tiny-cuda-nn for details
        desired_resolution = 4096
        base_grid_resolution = 16
        num_levels = 16
        per_level_scale = np.exp(np.log(desired_resolution / base_grid_resolution) / (num_levels-1))

        enc_cfg =  {
            "otype": "HashGrid",
            "n_levels": num_levels,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": base_grid_resolution,
            "per_level_scale" : per_level_scale
	    }

        gradient_scaling = 128.0
        self.encoder = tcnn.Encoding(3, enc_cfg)
        self.encoder.register_full_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] / gradient_scaling, ))

        # Setup MLP
        mlp_cfg = {
            "n_input_dims" : self.encoder.n_output_dims,
            "n_output_dims" : self.channels,
            "n_hidden_layers" : hidden,
            "n_neurons" : self.internal_dims
        }
        self.net = _MLP(mlp_cfg, gradient_scaling)
        print("Encoder output: %d dims" % (self.encoder.n_output_dims))

    # Sample texture at a given location
    def sample(self, texc):
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]
    
    # Sample texture at a given location
    @torch.no_grad()
    def sample_no_di(self, texc):
        _texc = (texc.view(-1, 3) - self.AABB[0][None, ...]) / (self.AABB[1][None, ...] - self.AABB[0][None, ...])
        _texc = torch.clamp(_texc, min=0, max=1)
        
        p_enc = self.encoder(_texc.contiguous())
        out = self.net.forward(p_enc)

        # Sigmoid limit and scale to the allowed range
        out = torch.sigmoid(out) * (self.min_max[1][None, :] - self.min_max[0][None, :]) + self.min_max[0][None, :]

        return out.view(*texc.shape[:-1], self.channels) # Remap to [n, h, w, c]

    # In-place clamp with no derivative to make sure values are in valid range after training
    def clamp_(self):
        pass

    def cleanup(self):
        tcnn.free_temporary_memory()

class EnvironmentLight:
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        self.mtx = None
        self.base = base

    def xfm(self, mtx):
        self.mtx = mtx

    def parameters(self):
        return [self.base]

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    @torch.no_grad()
    def generate_image(self, res):
        texcoord = pixel_grid(res[1], res[0])
        return dr.texture(self.base[None, ...].contiguous(), texcoord[None, ...].contiguous(), filter_mode='linear')[0]
    
def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):  
    base = torch.rand(base_res[0], base_res[1], 3, dtype=torch.float32, device='cuda') * scale + bias
    l = EnvironmentLight(base.clone().detach().requires_grad_(True))
    return l