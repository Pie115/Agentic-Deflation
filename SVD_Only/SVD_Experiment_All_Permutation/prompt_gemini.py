import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits
from permute_utils import apply_permutation

import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF

def create_matrix(amount_k, n=16, m=16, noise=1):  #create a synthetic matrix for our experiments
    matrix = np.zeros((n, m), dtype=float)
    for i in range(amount_k):
        u = np.random.randn(n)
        v = np.random.randn(m)
        s = np.random.uniform(10, 100)
        matrix += s * np.outer(u, v)
    if noise > 0:
        matrix += np.random.normal(0, noise, size=(n, m))
    matrix = np.rint(np.clip(matrix, 0, 255)).astype(int)
    return matrix

example = create_matrix(amount_k = 10)

def get_digits_image(index=0, dataset_name='digits', device='cpu'):
    if dataset_name == 'digits':
        data = load_digits()
        img = data.images[index]
        img = np.rint(img * (255.0 / 16.0)).astype(int)
        return img, data.target[index]
    elif dataset_name == 'cifar10':
        transform = transforms.ToTensor()
        cifar_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        idx = index % len(cifar_data)  #simple guard
        img_tensor, label = cifar_data[idx]
        img_tensor = TF.rgb_to_grayscale(img_tensor)
        img = img_tensor.squeeze(0).numpy()
        img = np.rint(img * 255.0).astype(int)
        return img, label
    
    elif dataset_name == 'synthetic':
        #choose k in [1..10] uniformly, this is how many rank 1 matrices we use to construct each sample
        amount_k = random.randint(1, 10)
        #use 16x16 by default for synthetic since it is in between 8x8 and 32x32
        img = create_matrix(amount_k=amount_k, n=16, m=16, noise=0)
        return img, amount_k
    
    elif dataset_name == 'synthetic_noisy':
        #If synthetic noisy is picked, we add random noise to the matrix along with components
        amount_k = random.randint(1, 10)
        #use 16x16 by default for synthetic since it is in between 8x8 and 32x32
        img = create_matrix(amount_k=amount_k, n=16, m=16, noise=1)
        return img, amount_k
    
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
        else:  #synthetic
            max_index = 1000000 

        while len(ex_list_unperm) < n_examples:
            ex_idx = random.randint(0, max_index)
            ex_mat, ex_num = get_digits_image(ex_idx, dataset_name=dataset_name)
            if ex_num in used_labels:
                continue
            used_labels.add(ex_num)

            if perm_method != 'none':
                if perm_method == 'group_n_teach':
                    ex_perm = apply_permutation(ex_mat, row_order, col_order)
                elif perm_method == 'sort':
                    if row_order is None or col_order is None:
                        row_order = np.argsort(-matrix.sum(axis=1))
                        col_order = np.argsort(-matrix.sum(axis=0))
                    ex_perm = ex_mat[np.ix_(row_order, col_order)]
                else:
                    ex_perm = ex_mat
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

def save_comparison_variable_examples(ex_list_unperm, original, gemini, truth, path="digits_rankk_comparison.png"):
    n_examples = len(ex_list_unperm)
    cols = n_examples + 3
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))

    col = 0
    for i in range(n_examples):
        axes[col].imshow(ex_list_unperm[i], cmap='gray', vmin=0, vmax=255)
        axes[col].set_title(f"Example {i+1}")
        axes[col].axis('off')
        col += 1

    axes[col].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[col].set_title("Original")
    axes[col].axis('off')
    col += 1

    axes[col].imshow(gemini, cmap='gray', vmin=0, vmax=255)
    axes[col].set_title("Gemini Rank k")
    axes[col].axis('off')
    col += 1

    axes[col].imshow(truth, cmap='gray', vmin=0, vmax=255)
    axes[col].set_title("True Rank k")
    axes[col].axis('off')

    fig.suptitle(f"ICL examples {n_examples}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def save_dual_comparison_variable_examples(
    ex_list_unperm, ex_list_perm,
    original, gemini, truth,
    original_blocky, gemini_blocky, truth_blocky,
    path="digits_rankk_comparison_dual.png"
):
    n_examples = len(ex_list_unperm)
    cols = n_examples + 3
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 6))

    col = 0
    for i in range(n_examples):
        axes[0, col].imshow(ex_list_unperm[i], cmap='gray', vmin=0, vmax=255)
        axes[0, col].set_title(f"Example {i+1}")
        axes[0, col].axis('off')

        axes[1, col].imshow(ex_list_perm[i], cmap='gray', vmin=0, vmax=255)
        axes[1, col].set_title(f"Example {i+1} blocky")
        axes[1, col].axis('off')
        col += 1

    axes[0, col].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0, col].set_title("Original")
    axes[0, col].axis('off')

    axes[1, col].imshow(original_blocky, cmap='gray', vmin=0, vmax=255)
    axes[1, col].set_title("Original blocky")
    axes[1, col].axis('off')
    col += 1

    axes[0, col].imshow(gemini, cmap='gray', vmin=0, vmax=255)
    axes[0, col].set_title("Gemini Rank k")
    axes[0, col].axis('off')

    axes[1, col].imshow(gemini_blocky, cmap='gray', vmin=0, vmax=255)
    axes[1, col].set_title("Gemini blocky")
    axes[1, col].axis('off')
    col += 1

    axes[0, col].imshow(truth, cmap='gray', vmin=0, vmax=255)
    axes[0, col].set_title("True Rank k")
    axes[0, col].axis('off')

    axes[1, col].imshow(truth_blocky, cmap='gray', vmin=0, vmax=255)
    axes[1, col].set_title("True blocky")
    axes[1, col].axis('off')

    fig.suptitle(f"ICL examples {n_examples}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()