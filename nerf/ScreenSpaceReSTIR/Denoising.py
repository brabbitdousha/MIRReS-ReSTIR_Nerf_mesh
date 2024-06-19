import torch
import numpy as np

def get_std(x, scale=1.0):
    r = torch.std(x[:,0:1])
    g = torch.std(x[:,1:2])
    b = torch.std(x[:,2:3])
    return scale * (r*r + g*g + b*b)

class EAWDenoise_run(torch.autograd.Function):
    @staticmethod
    def forward(ctx,    m,
                        c_phi, n_phi, p_phi,
                        framedim_x, framedim_y, stepWidth,
                        occ_map, color, normal_map, pos_map):
        out_color = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
        m.process_EAWDenoise(PHI=(c_phi, n_phi, p_phi),
        framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
        occ_map=occ_map, color=color, normal_map=normal_map, pos_map=pos_map,
        out_color = out_color
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        ctx.save_for_backward(occ_map, color, normal_map, pos_map, out_color)
        ctx.nums = [c_phi, n_phi, p_phi, framedim_x, framedim_y, stepWidth]
        ctx.slang_m = m

        return out_color
    @staticmethod
    def backward(ctx, grad_out_color):
        grad_out_color = grad_out_color.contiguous()
        (occ_map, color, normal_map, pos_map, out_color) = ctx.saved_tensors
        c_phi, n_phi, p_phi, framedim_x, framedim_y, stepWidth = ctx.nums
        m = ctx.slang_m
        grad_color = torch.zeros_like(color)
        grad_normal = torch.zeros_like(normal_map)
        grad_pos = torch.zeros_like(pos_map)
        m.process_EAWDenoise.bwd(PHI=(c_phi, n_phi, p_phi),
        framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
        occ_map=occ_map, color=(color, grad_color), normal_map=(normal_map, grad_normal), pos_map=(pos_map, grad_pos),
        out_color = (out_color, grad_out_color)
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))

        return (None,
                        None, None, None,
                        None, None, None,
                        None, grad_color, grad_normal, grad_pos)
    
def EAWDenoise_run_no_di(m,
                        c_phi, n_phi, p_phi,
                        framedim_x, framedim_y, stepWidth,
                        occ_map, color, normal_map, pos_map):
    out_color = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    m.process_EAWDenoise_no_di(PHI=(c_phi, n_phi, p_phi),
        framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
        occ_map=occ_map, color=color, normal_map=normal_map, pos_map=pos_map,
        out_color = out_color
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))
    return out_color

def EAWDenoise_run_phi(m,
                       phi,
                       framedim_x, framedim_y, stepWidth,
                       occ_map, color, normal_map, pos_map):
    out_color = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float, device='cuda')
    m.process_EAWDenoise_phi(PHI=phi,
    framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
    occ_map=occ_map, color=color, normal_map=normal_map, pos_map=pos_map,
    out_color = out_color
    )\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))
    return out_color

def EAWDenoise(m,
                       c_phi_scale, n_phi_scale, p_phi_scale,
                       stepWidth,
                       iter_time,
                       framedim_x, framedim_y,
                       occ_map, color, normal_map, pos_map
                       ):
    
    curr_color = color
    #'''
    indices = torch.where(occ_map>0.1)
    selected_color = curr_color.detach()
    selected_color = selected_color[indices[0],:]

    selected_normal = normal_map.detach()
    selected_normal = selected_normal[indices[0],:]

    selected_pos = pos_map.detach()
    selected_pos = selected_pos[indices[0],:]

    c_phi = get_std(selected_color, c_phi_scale)#0.49
    n_phi = get_std(selected_normal, n_phi_scale)#0.1
    p_phi = get_std(selected_pos, p_phi_scale)#0.00003
    #'''
    '''
    c_phi = c_phi*c_phi
    n_phi = n_phi*n_phi
    p_phi = p_phi*p_phi
    '''
    #phi = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float).cuda()
    for i in range(iter_time):
        '''
        m.process_variance(
        framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
        occ_map=occ_map, color=curr_color, normal_map=normal_map, pos_map=pos_map,
        out_phi = phi
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))
        '''
        #out_color = EAWDenoise_run_phi(m, phi, framedim_x, framedim_y, stepWidth, occ_map, curr_color, normal_map, pos_map)
        out_color = EAWDenoise_run.apply(m, c_phi, n_phi, p_phi, framedim_x, framedim_y, stepWidth, occ_map, curr_color, normal_map, pos_map)
        curr_color = out_color
        #recompute c_phi
        '''
        selected_color = curr_color.detach()
        selected_color = selected_color[indices[0],:]
        c_phi = get_std(selected_color, c_phi_scale)
        '''
        stepWidth /= 2
    
    return out_color

@torch.no_grad()
def Get_phi(
            c_phi_scale, n_phi_scale, p_phi_scale,
            occ_map, color, normal_map, pos_map):
    
    curr_color = color
    occ_map = occ_map.unsqueeze(-1)
    indices = torch.where(occ_map<=0.5)
    occ_map[indices[0],:] = 0
    #'''
    indices = torch.where(occ_map>0.1)
    selected_color = curr_color.detach()
    selected_color = selected_color[indices[0],:]

    selected_normal = normal_map.detach()
    selected_normal = selected_normal[indices[0],:]

    selected_pos = pos_map.detach()
    selected_pos = selected_pos[indices[0],:]

    c_phi = get_std(selected_color, c_phi_scale)#0.49
    n_phi = get_std(selected_normal, n_phi_scale)#0.1
    p_phi = get_std(selected_pos, p_phi_scale)#0.00003
    
    return c_phi, n_phi, p_phi

def EAWDenoise_use_phi(m,
                       c_phi, n_phi, p_phi,
                       stepWidth,
                       iter_time,
                       framedim_x, framedim_y,
                       occ_map, color, normal_map, pos_map
                       ):
    
    curr_color = color
    #'''
    indices = torch.where(occ_map>0.1)
    selected_color = curr_color.detach()
    selected_color = selected_color[indices[0],:]

    selected_normal = normal_map.detach()
    selected_normal = selected_normal[indices[0],:]

    selected_pos = pos_map.detach()
    selected_pos = selected_pos[indices[0],:]

    #'''
    '''
    c_phi = c_phi*c_phi
    n_phi = n_phi*n_phi
    p_phi = p_phi*p_phi
    '''
    #phi = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float).cuda()
    for i in range(iter_time):
        '''
        m.process_variance(
        framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
        occ_map=occ_map, color=curr_color, normal_map=normal_map, pos_map=pos_map,
        out_phi = phi
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))
        '''
        #out_color = EAWDenoise_run_phi(m, phi, framedim_x, framedim_y, stepWidth, occ_map, curr_color, normal_map, pos_map)
        out_color = EAWDenoise_run.apply(m, c_phi, n_phi, p_phi, framedim_x, framedim_y, stepWidth, occ_map, curr_color, normal_map, pos_map)
        curr_color = out_color
        #recompute c_phi
        '''
        selected_color = curr_color.detach()
        selected_color = selected_color[indices[0],:]
        c_phi = get_std(selected_color, c_phi_scale)
        '''
        stepWidth /= 2
    
    return out_color

@torch.no_grad()
def EAWDenoise_use_phi_no_di(m,
                       c_phi, n_phi, p_phi,
                       stepWidth,
                       iter_time,
                       framedim_x, framedim_y,
                       occ_map, color, normal_map, pos_map
                       ):
    
    curr_color = color
    #'''
    indices = torch.where(occ_map>0.1)
    selected_color = curr_color.detach()
    selected_color = selected_color[indices[0],:]

    selected_normal = normal_map.detach()
    selected_normal = selected_normal[indices[0],:]

    selected_pos = pos_map.detach()
    selected_pos = selected_pos[indices[0],:]

    #'''
    '''
    c_phi = c_phi*c_phi
    n_phi = n_phi*n_phi
    p_phi = p_phi*p_phi
    '''
    #phi = torch.zeros((framedim_x*framedim_y, 3), dtype=torch.float).cuda()
    for i in range(iter_time):
        '''
        m.process_variance(
        framedim_x=int(framedim_x), framedim_y=int(framedim_y), stepWidth=int(stepWidth),
        occ_map=occ_map, color=curr_color, normal_map=normal_map, pos_map=pos_map,
        out_phi = phi
        )\
        .launchRaw(blockSize=(16, 16, 1), gridSize=((int(framedim_x)+15)//16, (int(framedim_y)+15)//16, 1))
        '''
        #out_color = EAWDenoise_run_phi(m, phi, framedim_x, framedim_y, stepWidth, occ_map, curr_color, normal_map, pos_map)
        out_color = EAWDenoise_run_no_di(m, c_phi, n_phi, p_phi, framedim_x, framedim_y, stepWidth, occ_map, curr_color.detach(), normal_map.detach(), pos_map.detach())
        curr_color = out_color
        #recompute c_phi
        '''
        selected_color = curr_color.detach()
        selected_color = selected_color[indices[0],:]
        c_phi = get_std(selected_color, c_phi_scale)
        '''
        stepWidth /= 2
    
    return out_color