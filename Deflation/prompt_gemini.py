import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits
from permute_utils import apply_permutation

import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

def create_matrix(amount_k, n=16, m=16, noise=1):
    base = np.zeros((n, m), dtype=float)
    for i in range(amount_k):
        u = np.random.randn(n)
        v = np.random.randn(m)
        s = np.random.uniform(10, 100)
        base += s * np.outer(u, v)
    base_clip = np.rint(np.clip(base, 0, 255))
    if noise > 0:
        eps = np.random.normal(0, noise, size=(n, m))
        noisy = base + eps
    else:
        eps = np.zeros_like(base)
        noisy = base
    noisy_clip = np.rint(np.clip(noisy, 0, 255))
    noise_observed = noisy_clip - base_clip
    return noisy_clip.astype(int), base_clip.astype(int), noise_observed.astype(int)

def get_digits_image(index=0, dataset_name='digits', device='cpu'):
    if dataset_name == 'digits':
        data = load_digits()
        img = data.images[index]
        img = np.rint(img * (255.0 / 16.0)).astype(int)
        meta = {}
        return img, data.target[index], meta
    elif dataset_name == 'cifar10':
        transform = transforms.ToTensor()
        cifar_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        idx = index % len(cifar_data)
        img_tensor, label = cifar_data[idx]
        img_tensor = TF.rgb_to_grayscale(img_tensor)
        img = img_tensor.squeeze(0).numpy()
        img = np.rint(img * 255.0).astype(int)
        meta = {}
        return img, label, meta
    elif dataset_name == 'synthetic':
        amount_k = random.randint(1, 10)
        img, base_clip, noise_obs = create_matrix(amount_k=amount_k, n=16, m=16, noise=0)
        meta = {'k_true': amount_k, 'base': base_clip, 'noise_observed': noise_obs}
        return img, amount_k, meta
    elif dataset_name == 'synthetic_noisy':
        amount_k = random.randint(1, 10)
        img, base_clip, noise_obs = create_matrix(amount_k=amount_k, n=16, m=16, noise=1)
        meta = {'k_true': amount_k, 'base': base_clip, 'noise_observed': noise_obs}
        return img, amount_k, meta

def compute_svd_lists(matrix, k):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    u_list = [U[:, i].tolist() for i in range(k)]
    s_list = [float(S[i]) for i in range(k)]
    v_list = [VT[i, :].tolist() for i in range(k)]
    return u_list, s_list, v_list

def build_solver_prompt(matrix, true_number, k=3, perm_method='none', row_order=None, col_order=None, n_examples=2, dataset_name='digits'):
    prompt = (
        "Context:\n"
        "You are a linear algebra expert helping to analyze a matrix.\n\n"
        "Task:\n"
        f"Given a matrix, return the top {k} left singular vectors u1 to uk, "
        f"the top {k} right singular vectors v1 to vk, and the top {k} singular values s1 to sk from the matrix singular value decomposition.\n"
        f"That is, return u, s, and v such that A is approximately the sum over i from 1 to {k} of si times outer product of ui and vi.\n\n"
        "Instructions:\n"
        "Provide only the lists in this exact form with no prose\n"
        "u = [[...], [...], ...]\n"
        "s = [...]\n"
        "v = [[...], [...], ...]\n\n"
    )

    ex_list_unperm = []
    ex_list_perm = []

    if n_examples > 0:
        used_labels = set([true_number])
        if dataset_name == 'digits':
            max_index = 1796
        elif dataset_name == 'cifar10':
            max_index = 49999
        else:
            max_index = 1000000

        while len(ex_list_unperm) < n_examples:
            ex_idx = random.randint(0, max_index)
            ex_mat, ex_num, _meta = get_digits_image(ex_idx, dataset_name=dataset_name)
            if ex_num in used_labels:
                continue
            used_labels.add(ex_num)

            if perm_method != 'none':
                ex_perm = apply_permutation(ex_mat, row_order, col_order)
            else:
                ex_perm = ex_mat

            u_i, s_i, v_i = compute_svd_lists(ex_perm, k)
            prompt += f"Example {len(ex_list_unperm)+1}:\nMatrix:\n{ex_perm.tolist()}\n"
            prompt += f"u = {u_i}\n"
            prompt += f"s = {s_i}\n"
            prompt += f"v = {v_i}\n\n"

            ex_list_unperm.append(ex_mat)
            ex_list_perm.append(ex_perm)

    prompt += f"Matrix:\n{matrix.tolist()}"
    return prompt, ex_list_unperm, ex_list_perm

def parse_gemini_response(text):
    import ast
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    u_line = next(l for l in lines if l.startswith("u ="))
    s_line = next(l for l in lines if l.startswith("s ="))
    v_line = next(l for l in lines if l.startswith("v ="))
    u = np.array(ast.literal_eval(u_line.split("=", 1)[1].strip()))
    s = np.array(ast.literal_eval(s_line.split("=", 1)[1].strip()))
    v = np.array(ast.literal_eval(v_line.split("=", 1)[1].strip()))
    if u.ndim == 1:
        u = u[None, :]
    if v.ndim == 1:
        v = v[None, :]
    return u, s, v

def reconstruct_rankk_from_uvs(u, s, v):
    A = np.zeros((u.shape[1], v.shape[1]), dtype=float)
    for i in range(len(s)):
        A += s[i] * np.outer(u[i], v[i])
    return np.rint(np.clip(A, 0, 255)).astype(int)

def true_rankk(matrix, k=3):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    A = np.zeros_like(matrix, dtype=float)
    for i in range(k):
        A += S[i] * np.outer(U[:, i], VT[i, :])
    return np.rint(np.clip(A, 0, 255)).astype(int)

def save_single_image(img, path):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {path}")
