import os, imageio
import glob
import pyexr
import torch
import numpy as np
import scipy.signal
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

gt_dir = "./data/TensoIR_Synthetic/lego/"
albedo_dir = "./lego_11/validation_brdf/"
save_dir = "./ours_albedo/"
save_dir_gt = "./albedo_gt/"
save_gt = False
save_albedo = True
mask_thr = 0.9 # 0.3 for ficus and 0.9 for other scene

def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim

__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)

def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()

#hotdog
gt_list = []
test_list = []

gt_full_list = []
test_full_list = []
mask_list = []

print("load gt and test exrs----------")
for i in range(0,200):
    test_dir_idx = "000"+str(i)
    test_dir_idx = test_dir_idx[-3:]
    test_dir = "test_" + test_dir_idx
    gt_exr = pyexr.open(gt_dir+test_dir+"/diffuse-color.exr").get().astype(np.float64)  #diffuse-color
    mask = np.ones((800,800))
    mask[gt_exr[:,:,3]< mask_thr] = 0
    mask = mask.astype(int)
    gt_exr = gt_exr[...,:3] #* gt_exr[..., 3:4].repeat(3, axis=-1) + (1 - gt_exr[..., 3:4].repeat(3, axis=-1))
    if gt_exr.max()>1:
        print(i)
        raise
    gt_list.append(gt_exr[mask==1])
    gt_exr_full = gt_exr
    gt_exr_full[mask==0] = [1,1,1]
    gt_full_list.append(gt_exr_full)
    albedo_idx = "000"+str(i+1)  
    albedo_name = "ngp_stage1_ep0075_" + albedo_idx[-4:]+"_kd.exr"  #albedo
    test_exr = pyexr.open(albedo_dir+albedo_name).get().astype(np.float64)
    test_exr[mask==0] = [1,1,1]
    test_list.append(test_exr[mask==1])
    test_full_list.append(test_exr)
    mask_list.append(mask)
    #pyexr.write("./test_gt.exr",gt_exr_full)
    #pyexr.write("./test_albedo.exr",test_exr)

print("compute scale factor------------")
gt_array = np.concatenate(gt_list,axis=0)
test_array = np.concatenate(test_list,axis=0)
scale = np.median(gt_array / test_array.clip(min=1e-6),axis=0)
print(f'scale is : {scale}')
#albedo_array = test_array * scale

#print("compute psnr---------------")
#mse = np.mean((gt_array-albedo_array)**2)
#psnr = 20*np.log10(1.0/np.sqrt(mse))
#print("masked psnr:    ",psnr)

psnr_exr_list = []
psnr_png_list = []
ssim_list = []
lpips_vgg_list = []
lpips_alex_list = []

if save_albedo:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

if save_gt:
    if not os.path.exists(save_dir_gt):
        os.makedirs(save_dir_gt)

print("save scaled albedos and computing psnr----------")
for i in range(0,200):
    gt_img = gt_full_list[i]
    now_img = test_full_list[i]
    mask = mask_list[i]

    now_img[mask==1] = now_img[mask==1]*scale

    now_img = np.clip(now_img, 0, 1)
    gammaed_now_img = (now_img)**(1/2.2)
    gammaed_gt_img = (gt_img)**(1/2.2)

    gammaed_now_img_save = ((now_img)**(1/2.2) * 255).astype('uint8')
    gammaed_gt_img_save = ((gt_img)**(1/2.2) * 255).astype('uint8')
    
    #if save_albedo:
    #    pyexr.write(save_dir+f"scaled_kd_{i}.exr", now_img)
    #if save_gt:
    #    pyexr.write(save_dir_gt+f"gt_kd_{i}.exr",  gt_img)
    
    if save_albedo:
        imageio.imwrite(save_dir+f"gammaed_scaled_kd_{i}.png", gammaed_now_img_save)
    if save_gt:
        imageio.imwrite(save_dir_gt+f"gammaed_gt_kd_{i}.png", gammaed_gt_img_save) 

    loss_exr = np.mean((gt_img - now_img) ** 2)
    psnr_exr = -10.0 * np.log(loss_exr) / np.log(10.0)
    psnr_exr_list.append(psnr_exr)

    loss_png = np.mean((gammaed_gt_img.astype(np.float32) - gammaed_now_img.astype(np.float32)) ** 2)
    psnr_png = -10.0 * np.log(loss_png) / np.log(10.0)
    psnr_png_list.append(psnr_png)

    ssim = rgb_ssim(gammaed_now_img.astype(np.float32), gammaed_gt_img.astype(np.float32), 1)
    l_v = rgb_lpips(gammaed_gt_img.astype(np.float32), gammaed_now_img.astype(np.float32), 'vgg', 'cuda')
    l_a = rgb_lpips(gammaed_gt_img.astype(np.float32), gammaed_now_img.astype(np.float32), 'alex', 'cuda')


    ssim_list.append(ssim)
    lpips_vgg_list.append(l_v)
    lpips_alex_list.append(l_a)

    print(f'\r handling {i}  exr PSNR: {psnr_exr}  png PSNR: {psnr_png} ssim: {ssim} vgg lpips: {l_v} alex lpips: {l_a}')

total_psnr_exr = sum(psnr_exr_list)/len(psnr_exr_list)
total_png_exr = sum(psnr_png_list)/len(psnr_png_list)

total_ssim = sum(ssim_list)/len(ssim_list)
total_lpips = sum(lpips_vgg_list)/len(lpips_vgg_list)
total_a_lpips = sum(lpips_alex_list)/len(lpips_alex_list)

print("------final result----------------")
print(f'scale is : {scale}')
print(f'total exr psnr: {total_psnr_exr}  total png psnr: {total_png_exr} \nssim: {total_ssim} vgg lpips: {total_lpips} alex lpips: {total_a_lpips}')
    

    
    
    

