import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets import load_digits

def get_digits_image(index=0):
    data = load_digits()
    img = data.images[index]
    img = np.rint(img * (255.0 / 16.0)).astype(int)
    return img, data.target[index]

def compute_svd_lists(matrix, k):
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    u_list = [U[:, i].tolist() for i in range(k)]
    s_list = [float(S[i]) for i in range(k)]
    v_list = [VT[i, :].tolist() for i in range(k)]
    return u_list, s_list, v_list

def build_solver_prompt(matrix, true_number, k=3):
    prompt = (
        "Context:\n"
        "You are a linear algebra expert helping to analyze a matrix.\n\n"
        "Task:\n"
        f"Given a matrix, return the top {k} left singular vectors u₁, ..., uₖ, "
        f"the top {k} right singular vectors v₁, ..., vₖ, and the top {k} singular values s₁, ..., sₖ from the matrix's singular value decomposition (SVD).\n"
        f"That is, return u, s, and v such that A ≈ ∑ (sᵢ * uᵢ ⊗ vᵢᵗ) for i = 1 to {k}.\n\n"
        "Instructions:\n"
        "- Provide ONLY the lists, no prose:\n"
        "u = [[...], [...], ...]\n"
        "s = [...]\n"
        "v = [[...], [...], ...]\n\n"
    )

    # pick example indices different from the target digit label
    ex1_index = random.randint(0, 1796)
    ex1_matrix, ex1_number = get_digits_image(ex1_index)
    while true_number == ex1_number:
        ex1_index = random.randint(0, 1796)
        ex1_matrix, ex1_number = get_digits_image(ex1_index)

    ex2_index = random.randint(0, 1796)
    ex2_matrix, ex2_number = get_digits_image(ex2_index)
    while (true_number == ex2_number) or (ex1_number == ex2_number):
        ex2_index = random.randint(0, 1796)
        ex2_matrix, ex2_number = get_digits_image(ex2_index)

    # Use raw (unpermuted) examples
    u1, s1, v1 = compute_svd_lists(ex1_matrix, k)
    prompt += f"Example 1:\nMatrix:\n{ex1_matrix.tolist()}\n"
    prompt += f"u = {u1}\n"
    prompt += f"s = {s1}\n"
    prompt += f"v = {v1}\n\n"

    u2, s2, v2 = compute_svd_lists(ex2_matrix, k)
    prompt += f"Example 2:\nMatrix:\n{ex2_matrix.tolist()}\n"
    prompt += f"u = {u2}\n"
    prompt += f"s = {s2}\n"
    prompt += f"v = {v2}\n\n"

    # Target is also raw (unpermuted)
    prompt += f"Matrix:\n{matrix.tolist()}"
    return prompt, ex1_matrix, ex2_matrix

def parse_gemini_response(text):
    import ast
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    u_line = next(l for l in lines if l.startswith("u ="))
    s_line = next(l for l in lines if l.startswith("s ="))
    v_line = next(l for l in lines if l.startswith("v ="))
    u = np.array(ast.literal_eval(u_line.split("=",1)[1].strip()))
    s = np.array(ast.literal_eval(s_line.split("=",1)[1].strip()))
    v = np.array(ast.literal_eval(v_line.split("=",1)[1].strip()))
    if u.ndim == 1: u = u[None, :]
    if v.ndim == 1: v = v[None, :]
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

def save_comparison_with_examples(ex1, ex2, original, gemini, truth, path="digits_rankk_comparison.png"):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    images = [ex1, ex2, original, gemini, truth]
    titles = ["Example 1", "Example 2", "Original", "Gemini Rank-k", "True Rank-k"]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

def save_single_image(img, path):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved {path}")
