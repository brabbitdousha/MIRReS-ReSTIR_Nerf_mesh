import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from .render_helper import *

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, geom_init=False, weight_norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.geom_init = geom_init

        net = []
        for l in range(num_layers):

            in_dim = self.dim_in if l == 0 else self.dim_hidden
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden

            net.append(nn.Linear(in_dim, out_dim, bias=bias))
        
            if geom_init:
                if l == num_layers - 1:
                    torch.nn.init.normal_(net[l].weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-4)
                    if bias: torch.nn.init.constant_(net[l].bias, -0.5) # sphere init (very important for hashgrid encoding!)

                elif l == 0:
                    torch.nn.init.normal_(net[l].weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    torch.nn.init.constant_(net[l].weight[:, 3:], 0.0)
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)

                else:
                    torch.nn.init.normal_(net[l].weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)
            
            if weight_norm:
                net[l] = nn.utils.weight_norm(net[l])

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.geom_init:
                    x = F.softplus(x, beta=100)
                else:
                    x = F.relu(x, inplace=True)
        return x


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 specular_dim=3,
                 ):

        super().__init__(opt)
        
        #----------------original ngp part
        num_layers=2
        hidden_dim=64
        geo_feat_dim=15
        num_layers_color=3
        hidden_dim_color=64

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder("hashgrid_tcnn" if self.opt.tcnn else "hashgrid", level_dim=2, desired_resolution=2048 * self.bound, interpolation='linear')

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder('sh')
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
        #----------------

        #brdf part
        if self.opt.stage == 1 and self.opt.use_brdf:
            kd_min, kd_max = torch.tensor(self.opt.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(self.opt.kd_max, dtype=torch.float32, device='cuda')
            ks_min, ks_max = torch.tensor(self.opt.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(self.opt.ks_max, dtype=torch.float32, device='cuda')
            mlp_min = torch.cat((kd_min[0:3], ks_min), dim=0)
            mlp_max = torch.cat((kd_max[0:3], ks_max), dim=0)
            real_bound = self.opt.bound
            aabb_train = torch.FloatTensor([-real_bound, -real_bound, -real_bound, real_bound, real_bound, real_bound])
            self.mlp_mat_opt = MLPTexture3D(aabb_train, channels=6, min_max=[mlp_min, mlp_max])
            self.lgt = create_trainable_env_rnd(self.opt.light_probe_res_hw, scale=0.0, bias=0.5)
            #only for dump render----
            if not self.opt.use_restir:
                envmap_h = self.opt.light_probe_res_hw[0]
                envmap_w = self.opt.light_probe_res_hw[1]
                self.light_area_weight, self.fixed_viewdirs = generate_envir_map_dir(envmap_h, envmap_w)
            #--------------

            #for relighting
            if self.opt.test and self.opt.envmap_path != 'None':
                env_map0 = cv2.imread(self.opt.envmap_path,flags = cv2.IMREAD_ANYDEPTH)
                env_map0 = cv2.cvtColor(env_map0, cv2.COLOR_BGR2RGB)
                env_map0 = torch.tensor(env_map0, device="cuda")
                self.env_map0 = env_map0
            

        # sdf
        if self.opt.sdf:
            self.register_parameter('variance', nn.Parameter(torch.tensor(0.3, dtype=torch.float32)))

    def forward(self, x, d, c=None, shading='full'):
        
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        h = self.encoder(x, bound=self.bound)

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color, torch.zeros_like(color)


    def density(self, x):

        # sigma
        h = self.encoder(x, bound=self.bound)
        #h = x #torch.cat([x, h], dim=-1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        results = {}

        sigma = trunc_exp(h[..., 0])

        results['sigma'] = sigma

        return results

    # init the sdf to two spheres by pretraining, assume view cameras fall between the spheres
    def init_double_sphere(self, r1=0.5, r2=1.5, iters=8192, batch_size=8192):
        # sphere init is only for sdf mode!
        if not self.opt.sdf:
            return
        # import kiui
        import tqdm
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(list(self.parameters()), lr=1e-3)
        pbar = tqdm.trange(iters, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        for _ in range(iters):
            # random points inside [-b, b]^3
            xyzs = torch.rand(batch_size, 3, device='cuda') * 2 * self.bound - self.bound
            d = torch.norm(xyzs, p=2, dim=-1)
            gt_sdf = torch.where(d < (r1 + r2) / 2, d - r1, r2 - d)
            # kiui.lo(xyzs, gt_sdf)
            pred_sdf = self.density(xyzs)['sigma']
            loss = loss_fn(pred_sdf, gt_sdf)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'pretrain sdf loss={loss.item():.8f}')
            pbar.update(1)
    
    # finite difference
    def normal(self, x, epsilon=1e-4):

        if self.opt.tcnn:
            with torch.enable_grad():
                x.requires_grad_(True)
                sigma = self.density(x)['sigma']
                normal = torch.autograd.grad(torch.sum(sigma), x, create_graph=True)[0] # [N, 3]
        else:
            dx_pos = self.density((x + torch.tensor([[epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dx_neg = self.density((x + torch.tensor([[-epsilon, 0.00, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dy_pos = self.density((x + torch.tensor([[0.00, epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dy_neg = self.density((x + torch.tensor([[0.00, -epsilon, 0.00]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dz_pos = self.density((x + torch.tensor([[0.00, 0.00, epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            dz_neg = self.density((x + torch.tensor([[0.00, 0.00, -epsilon]], device=x.device)).clamp(-self.bound, self.bound))['sigma']
            
            normal = torch.stack([
                0.5 * (dx_pos - dx_neg) / epsilon, 
                0.5 * (dy_pos - dy_neg) / epsilon, 
                0.5 * (dz_pos - dz_neg) / epsilon
            ], dim=-1)

        return normal
    
    def geo_feat(self, x, c=None):

        geo_feat = torch.zeros((x.shape[0], 6), dtype=torch.float, device='cuda')

        return geo_feat

    def rgb(self, x, d, c=None, shading='full'):

        # sigma
        h = self.encoder(x, bound=self.bound)

        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        #sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        
        return color, torch.zeros_like(color)


    # optimizer utils
    def get_params(self, lr):

        params = super().get_params(lr)
        params.extend([
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ])

        if self.opt.sdf:
            params.append({'params': self.variance, 'lr': lr * 0.1})

        return params
    
    def get_mat_params(self):
        params = []
        params.append({'params': self.mlp_mat_opt.parameters(), 'lr': self.opt.learning_rate_mat})
        return params
    
    def get_light_params(self):
        params = []
        params.append({'params': self.lgt.parameters(), 'lr': self.opt.learning_rate_lgt})
        return params