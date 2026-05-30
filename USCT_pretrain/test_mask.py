import skimage.draw, h5py, skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torch

def scale2uint16(d_img):
    _min, _max = d_img.min(), d_img.max()
    if _max == _min:
        r_img = d_img[:] - _max
    else:
        r_img = (d_img[:] - _min) * 65535. / (_max - _min)
    return r_img.astype(np.uint16)

def tomo_sim_rnd(h, w, nproj):
    image = skimage.draw.random_shapes(image_shape=(h, w), max_shapes=30, min_shapes=10, \
                                       min_size=w//10, max_size=w//4, intensity_range=(80, 200),\
                                       channel_axis=None, allow_overlap=False)[0]
    rr, cc = skimage.draw.disk((h/2, w/2), w/2)
    mask   = np.zeros((h, w), dtype=np.uint8)
    mask[rr, cc] = 1
    image *= mask

    theta = np.linspace(0., 180., nproj, endpoint=False)
    sino  = skimage.transform.radon(image, theta=theta, circle=True)

    return image, scale2uint16(sino.T)


def random_masking(x, mask_ratio):
    print(x.shape)
    N, L, D = x.shape  # batch, length, dim (a.k.a. BNC)
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

image,sino = tomo_sim_rnd(256, 256, 180)

print(sino.shape)
# print(sino)

# 可视化图像
plt.imshow(image, cmap='gray')
plt.title('Random Shapes Image')
plt.axis('off')  # 隐藏坐标轴
plt.show()

plt.imshow(sino, cmap='gray')
plt.title('Random Shapes Image')
plt.axis('off')  # 隐藏坐标轴
plt.show()


sino = torch.from_numpy(sino.astype(np.int64)).cuda()

#print(sino.shape)
#print(sino.squeeze(0).shape)
x_masked, mask, ids_restore = random_masking(sino.unsqueeze(0), 0.8)  # 注意这里squeeze需要加括号
print(x_masked.shape,mask.shape,ids_restore.shape)
print(mask)
print(ids_restore)
plt.imshow(x_masked[0].cpu().numpy(), cmap='gray')
plt.show()
plt.savefig('masked_image.png', dpi=300, bbox_inches='tight')
plt.close()


