import torch
import numpy as np

def make_sampleable(m, env_map, width, height):
    env_map = env_map.contiguous()
    weight = torch.zeros([width*height,1],dtype=torch.float, device='cuda')
    m.make_sampleable(env_tex=env_map, weight=weight, width=int(width), height=int(height))\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(height)+15)//16, (int(width)+15)//16, 1))

    #Distribution2D
    cdf_ = torch.zeros([height, 1],dtype=torch.float, device='cuda')
    mcdf_ = torch.zeros([height+1,1],dtype=torch.float, device='cuda')

    pdf_ = weight.reshape(height, width, 1)
    mpdf_ = pdf_.sum(1)
    cdf_ = torch.cat([cdf_,pdf_.cumsum(1).reshape(height, width)],dim=-1)
    mcdf_[1:] = mpdf_.cumsum(0)
    cdf_ = cdf_.reshape(-1,1)

    m.Distribution2D(w=int(width), h=int(height), pdf_=weight, cdf_=cdf_)\
    .launchRaw(blockSize=(16, 16, 1), gridSize=((int(height)+15)//16, (int(width)+15)//16, 1))
    cdf_ = cdf_.reshape(height, width+1, 1)
    cdf_[:, -1, 0] = 1.
    cdf_ = cdf_.reshape(-1,1)
    total_weight = mcdf_[-1]
    mpdf_ = mpdf_/total_weight
    mcdf_ = mcdf_/total_weight
    mcdf_[-1] = 1.
    return weight, cdf_, mpdf_, mcdf_

def GenerateLightTiles(m,
                       debug_out,
                       env_tex, pdf_, cdf_, mpdf_, mcdf_,
                       width, height,
                       frameIndex,
                       light_data,
                       light_uv,
                       light_inv_pdf,
                       light_tile_count = 128,
                       light_tile_size = 1024):

    m.process_GenerateLightTiles(env_tex=env_tex, pdf_=pdf_, cdf_=cdf_, mpdf_=mpdf_, mcdf_=mcdf_,
                                width=int(width), height=int(height),
                                frameIndex=int(frameIndex),
                                light_data=light_data,
                                light_uv=light_uv, 
                                light_inv_pdf=light_inv_pdf,
                                debug_out=debug_out
                                                    )\
    .launchRaw(blockSize=(256, 1, 1), gridSize=(light_tile_size, light_tile_count, 1))

    return 'hello'
