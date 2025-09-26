import numpy as np
import matplotlib.pyplot as plt
import os
import random
from skimage.metrics import structural_similarity as ssim
from permute_utils import groupnteach_permutation

np.random.seed(0)
random.seed(0)

k = 1
n_per_method = 200
perm_methods = ['none', 'sort', 'group_n_teach']
k_blocks_default = 4

dataset_name = 'digits'  # digits or cifar10 or synthetic or synthetic_noisy

if dataset_name == 'digits':
    max_index = 1796
elif dataset_name == 'cifar10':
    max_index = 49999
else:
    max_index = 1000000

all_indices = random.sample(range(1, max_index + 1), n_per_method)

def mean_std(arr):
    arr = np.array(arr, dtype=float)
    if arr.size == 0:
        return float('nan'), float('nan')
    return float(arr.mean()), float(arr.std(ddof=0))

def rmse(a, b):
    a = a.astype(float); b = b.astype(float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def true_rank1(A):
    U, S, VT = np.linalg.svd(A.astype(float), full_matrices=False)
    R = np.outer(U[:, 0] * S[0], VT[0, :])
    return np.rint(np.clip(R, 0, 255)).astype(int), float(S[0])

def random_rank1_same_scale(n, m, s, seed):
    rng = np.random.RandomState(seed)
    u = rng.randn(n)
    v = rng.randn(m)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu > 0: u = u / nu
    if nv > 0: v = v / nv
    R = s * np.outer(u, v)
    return np.rint(np.clip(R, 0, 255)).astype(int)

# deterministic getters so samples are identical across perm methods
def get_sample(index, dataset):
    if dataset == 'digits':
        # sklearn load is deterministic by index
        from sklearn.datasets import load_digits
        data = load_digits()
        img = np.rint(data.images[index] * (255.0 / 16.0)).astype(int)
        lbl = int(data.target[index])
        return img, lbl
    elif dataset == 'cifar10':
        import torch
        from torchvision import datasets, transforms
        import torchvision.transforms.functional as TF
        transform = transforms.ToTensor()
        cifar = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        idx = index % len(cifar)
        img_tensor, lbl = cifar[idx]
        img_tensor = TF.rgb_to_grayscale(img_tensor)
        img = np.rint(img_tensor.squeeze(0).numpy() * 255.0).astype(int)
        return img, int(lbl)
    elif dataset in ('synthetic', 'synthetic_noisy'):
        n = 16; m = 16
        rng = np.random.RandomState(123456 + index)
        k_true = rng.randint(1, 11)
        base = np.zeros((n, m), dtype=float)
        for _ in range(k_true):
            u = rng.randn(n); v = rng.randn(m); s = rng.uniform(10, 100)
            base += s * np.outer(u, v)
        base_clip = np.rint(np.clip(base, 0, 255))
        if dataset == 'synthetic_noisy':
            eps = rng.normal(0, 1.0, size=(n, m))
            noisy = base + eps
        else:
            noisy = base
        noisy_clip = np.rint(np.clip(noisy, 0, 255)).astype(int)
        return noisy_clip, int(k_true)
    else:
        raise ValueError("unknown dataset")

# precompute and cache the exact samples once
cached_samples = []
for idx in all_indices:
    img, lbl = get_sample(idx, dataset_name)
    cached_samples.append((idx, img, lbl))

def run_random_baseline_for_method(perm_method, samples, k_blocks=k_blocks_default, dataset_name='digits'):
    rand_vs_orig_rmse = []
    rand_vs_orig_ssim = []

    rand_vs_svd_rmse = []
    rand_vs_svd_ssim = []

    svd_vs_orig_rmse = []
    svd_vs_orig_ssim = []

    for (idx, img, lbl) in samples:
        if perm_method == 'group_n_teach':
            img_perm_float, row_order, col_order, row_inv, col_inv, groups, D = groupnteach_permutation(
                img.astype(float), k=k_blocks, density_mode="mean", random_state=0
            )
            img_perm = img_perm_float.astype(int)
        elif perm_method == 'sort':
            row_order = np.argsort(-img.sum(axis=1))
            col_order = np.argsort(-img.sum(axis=0))
            img_perm = img[np.ix_(row_order, col_order)]
        else:
            img_perm = img

        A_svd, s_top = true_rank1(img)
        A_rand = random_rank1_same_scale(img.shape[0], img.shape[1], s_top, seed=0 + idx)

        try:
            ssim_r_o = ssim(img.astype(float), A_rand.astype(float), data_range=255)
            ssim_r_s = ssim(A_rand.astype(float), A_svd.astype(float), data_range=255)
            ssim_s_o = ssim(img.astype(float), A_svd.astype(float), data_range=255)

            rmse_r_o = rmse(A_rand, img)
            rmse_r_s = rmse(A_rand, A_svd)
            rmse_s_o = rmse(A_svd, img)
        except ValueError:
            continue

        rand_vs_orig_rmse.append(rmse_r_o)
        rand_vs_orig_ssim.append(ssim_r_o)

        rand_vs_svd_rmse.append(rmse_r_s)
        rand_vs_svd_ssim.append(ssim_r_s)

        svd_vs_orig_rmse.append(rmse_s_o)
        svd_vs_orig_ssim.append(ssim_s_o)

    dest_folder = f"final_results_random_svd_{dataset_name}_{perm_method}"
    os.makedirs(dest_folder, exist_ok=True)
    metrics_path = os.path.join(dest_folder, "metrics.txt")

    r_o_rmse_m, r_o_rmse_s = mean_std(rand_vs_orig_rmse)
    r_o_ssim_m, r_o_ssim_s = mean_std(rand_vs_orig_ssim)

    r_s_rmse_m, r_s_rmse_s = mean_std(rand_vs_svd_rmse)
    r_s_ssim_m, r_s_ssim_s = mean_std(rand_vs_svd_ssim)

    s_o_rmse_m, s_o_rmse_s = mean_std(svd_vs_orig_rmse)
    s_o_ssim_m, s_o_ssim_s = mean_std(svd_vs_orig_ssim)

    with open(metrics_path, "w") as f:
        f.write(f"Permutation method {perm_method}\n")
        f.write(f"Samples {len(samples)}\n\n")

        f.write("Random vs Original\n")
        f.write(f"RMSE {r_o_rmse_m:.6f} ± {r_o_rmse_s:.6f}\n")
        f.write(f"SSIM {r_o_ssim_m:.6f} ± {r_o_ssim_s:.6f}\n\n")

        f.write("Random vs NumPy Rank-1\n")
        f.write(f"RMSE {r_s_rmse_m:.6f} ± {r_s_rmse_s:.6f}\n")
        f.write(f"SSIM {r_s_ssim_m:.6f} ± {r_s_ssim_s:.6f}\n\n")

        f.write("NumPy Rank-1 vs Original\n")
        f.write(f"RMSE {s_o_rmse_m:.6f} ± {s_o_rmse_s:.6f}\n")
        f.write(f"SSIM {s_o_ssim_m:.6f} ± {s_o_ssim_s:.6f}\n")

    print(f"[{dataset_name} | {perm_method}] wrote metrics -> {metrics_path}")

for m in perm_methods:
    run_random_baseline_for_method(m, cached_samples, k_blocks=k_blocks_default, dataset_name=dataset_name)