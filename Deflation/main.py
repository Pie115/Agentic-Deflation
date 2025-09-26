import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from skimage.metrics import structural_similarity as ssim
import random
from prompt_qwen import load_qwen_vl, evaluate_pair_with_qwen, evaluate_deflation_with_qwen
from prompt_gemini import (
    get_digits_image, compute_svd_lists, build_solver_prompt, parse_gemini_response,
    reconstruct_rankk_from_uvs, true_rankk, save_single_image
)
import shutil
from permute_utils import groupnteach_permutation
import time

def save_deflation_trajectory(unpermuted_steps, permuted_steps, path="deflation_trajectory.png"):
    assert len(unpermuted_steps) == len(permuted_steps) and len(permuted_steps) >= 1
    steps = len(permuted_steps)
    fig, axes = plt.subplots(2, steps, figsize=(3 * steps, 6))
    if steps == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for j in range(steps):
        ax_top = axes[0, j]
        ax_top.imshow(unpermuted_steps[j], cmap='gray', vmin=0, vmax=255)
        ax_top.set_title(f"Step {j}")
        ax_top.axis('off')
        ax_bot = axes[1, j]
        ax_bot.imshow(permuted_steps[j], cmap='gray', vmin=0, vmax=255)
        ax_bot.set_title(f"Step {j} (perm)")
        ax_bot.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

np.random.seed(0)
random.seed(0)

k = 1
n_per_dataset = 200
perm_methods = ['none']  # can be none, sort, or group_n_teach
icl_examples_list = [2]
k_blocks_default = 4  # 2 for digits, 8 for cifar10, 4 for synthetic

dataset_name = 'synthetic'  # can be digits or cifar10 or synthetic or synthetic_noisy

if dataset_name == 'digits':
    max_index = 1796
elif dataset_name == 'cifar10':
    max_index = 49999
else:
    max_index = 1000000

indexes = random.sample(range(1, max_index + 1), n_per_dataset)

genai.configure(api_key="PUT API KEY HERE")
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

qwen_model, processor = load_qwen_vl()

def apply_perm_for_method(img, perm_method, k_blocks=4):
    n, m = img.shape
    meta = {}
    if perm_method == 'group_n_teach':
        img_perm_float, row_order, col_order, row_inv, col_inv, groups, D = groupnteach_permutation(
            img.astype(float), k=k_blocks, density_mode="mean", random_state=0
        )
        img_perm = img_perm_float.astype(int)
        meta.update({"groups": groups, "D": D})
        return img_perm, row_order, col_order, row_inv, col_inv, meta
    elif perm_method == 'sort':
        row_order = np.argsort(-img.sum(axis=1))
        col_order = np.argsort(-img.sum(axis=0))
        row_inv = np.argsort(row_order)
        col_inv = np.argsort(col_order)
        img_perm = img[np.ix_(row_order, col_order)]
        return img_perm.astype(int), row_order, col_order, row_inv, col_inv, meta
    else:
        row_order = np.arange(n)
        col_order = np.arange(m)
        row_inv = row_order
        col_inv = col_order
        img_perm = img
        return img_perm.astype(int), row_order, col_order, row_inv, col_inv, meta

def fro_norm(A):
    return float(np.sqrt(np.sum((A.astype(float))**2)))

def numpy_deflate_to_fraction(img_perm, frac=0.1):
    A = img_perm.astype(float).copy()
    A = np.clip(A, 0, None)
    A0 = A.copy()
    norm0 = fro_norm(A0)
    steps = 0
    while norm0 > 0 and fro_norm(A) > frac * norm0:
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        rank1 = np.outer(U[:, 0] * S[0], VT[0, :])
        A = np.clip(A - rank1, 0, None)
        steps += 1
        if steps > 256:
            break
    return np.rint(np.clip(A, 0, 255)).astype(int), steps

def numpy_deflate_k_steps(img_perm, k_steps):
    A = img_perm.astype(float).copy()
    A = np.clip(A, 0, None)
    steps = 0
    for _ in range(max(0, int(k_steps))):
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        rank1 = np.outer(U[:, 0] * S[0], VT[0, :])
        A = np.clip(A - rank1, 0, None)
        steps += 1
        if steps > 256:
            break
    return np.rint(np.clip(A, 0, 255)).astype(int), steps

def rmse(a, b):
    a = a.astype(float); b = b.astype(float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def run_deflation_for_method(perm_method, idx_list, k_blocks=k_blocks_default, n_examples=2, dataset_name='digits'):
    print(f"\n=== Running deflation with perm_method = {perm_method}, icl={n_examples} on {dataset_name} ===")
    rmse_list = []
    nrmse_list = []
    steps_agent_list = []
    steps_numpy_list = []

    rmse_numpy_list = []
    nrmse_numpy_list = []

    dest_root = f"final_results_deflation_{dataset_name}_{perm_method}_icl{n_examples}"
    os.makedirs(dest_root, exist_ok=True)

    for idx_pos, idx_value in enumerate(idx_list):
        img, number, meta = get_digits_image(idx_value, dataset_name=dataset_name)

        img_perm, row_order, col_order, row_inv, col_inv, _meta = apply_perm_for_method(
            img, perm_method, k_blocks=k_blocks
        )

        save_single_image(img_perm, "original_blocky_no_deflation.png")

        permuted_steps = [img_perm.copy()]
        unpermuted_steps = [img.copy()]

        deflation_decision = 'accept'

        while deflation_decision == 'accept':
            prompt, ex_list_unperm, ex_list_perm = build_solver_prompt(
                img_perm, number, k=k, perm_method=perm_method,
                row_order=row_order, col_order=col_order,
                n_examples=n_examples, dataset_name=dataset_name
            )

            try:
                resp = gemini_model.generate_content([prompt])
            except Exception as e:
                print(f"Gemini API error {type(e).__name__}: {e}. Sleeping 10s and retrying...")
                time.sleep(10)
                try:
                    resp = gemini_model.generate_content([prompt])
                except Exception as e2:
                    print(f"Repeated Gemini API error: {e2}. Skipping this sample.")
                    deflation_decision = 'reject'
                    break

            print("========================Prompt=========================")
            print(prompt)
            print("========================Solver Response=========================")
            print("Gemini output:\n", getattr(resp, "text", ""))

            gemini_decision = 'reject'
            max_gemini_retries = 10
            gemini_attempts = 0

            while gemini_decision == 'reject':
                try:
                    u_pred, s_pred, v_pred = parse_gemini_response(resp.text)
                    A_gemini_perm = reconstruct_rankk_from_uvs(u_pred, s_pred, v_pred)
                except Exception as e:
                    print("Parse/reconstruct error:", e)
                    A_gemini_perm = None

                if A_gemini_perm is None or A_gemini_perm.shape != img_perm.shape:
                    print("Shape mismatch or invalid reconstruction; retrying Gemini...")
                    gemini_attempts += 1
                    if gemini_attempts > max_gemini_retries:
                        print("Max Gemini retries reached; skipping this sample.")
                        deflation_decision = 'reject'
                        break
                    resp = gemini_model.generate_content([prompt])
                    continue

                A_true_blocky = true_rankk(img_perm, k=k)
                try:
                    ssim_gemini = ssim(img_perm.astype(float), A_gemini_perm.astype(float), data_range=255)
                    ssim_gemini_vs_true = ssim(A_gemini_perm.astype(float), A_true_blocky.astype(float), data_range=255)
                    rmse_gemini_vs_true = np.sqrt(np.mean((A_gemini_perm.astype(float) - A_true_blocky.astype(float)) ** 2))
                except ValueError as e:
                    print(f"SSIM computation error: {e}")
                    gemini_attempts += 1
                    if gemini_attempts > max_gemini_retries:
                        print("Max Gemini retries reached after SSIM errors; skipping this sample.")
                        deflation_decision = 'reject'
                        break
                    resp = gemini_model.generate_content([prompt])
                    continue

                print("========================SSIM Values=========================")
                print(f"SSIM (Gemini(blocky) vs Original(blocky)): {ssim_gemini:.4f}")
                print(f"SSIM (Gemini(blocky) vs True(blocky)): {ssim_gemini_vs_true:.4f}")
                print(f"RMSE (Gemini vs True SVD): {rmse_gemini_vs_true:.4f}")
                nrmse_255 = rmse_gemini_vs_true / 255.0
                print(f"NRMSE (÷255): {nrmse_255:.6f}")

                save_single_image(img_perm, "original_blocky.png")
                save_single_image(A_gemini_perm, "gemini_rankk_only.png")
                result = evaluate_pair_with_qwen(
                    qwen_model, processor,
                    img_path_true="original_blocky.png",
                    img_path_gemini="gemini_rankk_only.png",
                )

                print("========================SVD Evaluator Response=========================")
                print(result)
                if os.path.exists('gemini_rankk_only.png'):
                    os.remove('gemini_rankk_only.png')
                if os.path.exists('original_blocky.png'):
                    os.remove('original_blocky.png')

                if result["decision"] == 'reject':
                    resp = gemini_model.generate_content([prompt])
                    gemini_attempts += 1
                    if gemini_attempts > max_gemini_retries:
                        print("Max Gemini retries reached after evaluator rejection; skipping this sample.")
                        deflation_decision = 'reject'
                        break
                else:
                    gemini_decision = 'accept'

            if deflation_decision == 'reject':
                traj_path = os.path.join(dest_root, f"deflation_trajectory_{idx_value}.png")
                save_deflation_trajectory(unpermuted_steps, permuted_steps, path=traj_path)
                continue

            img_perm = np.clip(img_perm - A_gemini_perm, 0, None)
            save_single_image(img_perm, "deflated_blocky.png")

            unpermuted_residual = img_perm[np.ix_(row_inv, col_inv)]
            permuted_steps.append(img_perm.copy())
            unpermuted_steps.append(unpermuted_residual.copy())

            deflation_result = evaluate_deflation_with_qwen(
                qwen_model, processor,
                img_path_true="original_blocky_no_deflation.png",
                img_path_gemini="deflated_blocky.png",
            )

            print("========================Deflation Evaluator Response=========================")
            print(deflation_result)

            if deflation_result["decision"] == 'reject':
                deflation_decision = 'reject'
                traj_path = os.path.join(dest_root, f"deflation_trajectory_{idx_value}.png")
                save_deflation_trajectory(unpermuted_steps, permuted_steps, path=traj_path)
                break

        agent_residual_perm = img_perm.copy()
        steps_agent = max(0, len(permuted_steps) - 1)
        steps_agent_list.append(steps_agent)

        if dataset_name == 'synthetic':
            target_perm = np.zeros_like(agent_residual_perm)
            k_true = int(meta.get('k_true', number))
            numpy_residual_perm, steps_numpy = numpy_deflate_k_steps(permuted_steps[0], k_true)
            steps_numpy_list.append(steps_numpy)
            rmse_numpy = rmse(numpy_residual_perm, target_perm)
            nrmse_numpy = rmse_numpy / 255.0
            rmse_numpy_list.append(rmse_numpy)
            nrmse_numpy_list.append(nrmse_numpy)
        elif dataset_name == 'synthetic_noisy':
            noise_obs = meta.get('noise_observed', np.zeros_like(agent_residual_perm))
            if perm_method != 'none':
                noise_obs = noise_obs[np.ix_(row_order, col_order)]
            target_perm = noise_obs
            k_true = int(meta.get('k_true', number))
            numpy_residual_perm, steps_numpy = numpy_deflate_k_steps(permuted_steps[0], k_true)
            steps_numpy_list.append(steps_numpy)
            rmse_numpy = rmse(numpy_residual_perm, target_perm)
            nrmse_numpy = rmse_numpy / 255.0
            rmse_numpy_list.append(rmse_numpy)
            nrmse_numpy_list.append(nrmse_numpy)
        else:
            base_residual_perm, steps_numpy = numpy_deflate_to_fraction(permuted_steps[0], frac=0.1)
            target_perm = base_residual_perm
            steps_numpy_list.append(steps_numpy)
            rmse_numpy = rmse(base_residual_perm, target_perm)
            nrmse_numpy = rmse_numpy / 255.0
            rmse_numpy_list.append(rmse_numpy)
            nrmse_numpy_list.append(nrmse_numpy)

        e_rmse = rmse(agent_residual_perm, target_perm)
        e_nrmse = e_rmse / 255.0
        rmse_list.append(e_rmse)
        nrmse_list.append(e_nrmse)

    avg_rmse = float(np.mean(rmse_list)) if len(rmse_list) > 0 else float('nan')
    avg_nrmse = float(np.mean(nrmse_list)) if len(nrmse_list) > 0 else float('nan')
    avg_steps_agent = float(np.mean(steps_agent_list)) if len(steps_agent_list) > 0 else 0.0
    avg_steps_numpy = float(np.mean(steps_numpy_list)) if len(steps_numpy_list) > 0 else 0.0
    avg_rmse_numpy = float(np.mean(rmse_numpy_list)) if len(rmse_numpy_list) > 0 else float('nan')
    avg_nrmse_numpy = float(np.mean(nrmse_numpy_list)) if len(nrmse_numpy_list) > 0 else float('nan')

    std_rmse = float(np.std(rmse_list, ddof=0)) if len(rmse_list) > 0 else float('nan')
    std_nrmse = float(np.std(nrmse_list, ddof=0)) if len(nrmse_list) > 0 else float('nan')
    std_steps_agent = float(np.std(steps_agent_list, ddof=0)) if len(steps_agent_list) > 0 else float('nan')
    std_steps_numpy = float(np.std(steps_numpy_list, ddof=0)) if len(steps_numpy_list) > 0 else float('nan')
    std_rmse_numpy = float(np.std(rmse_numpy_list, ddof=0)) if len(rmse_numpy_list) > 0 else float('nan')
    std_nrmse_numpy = float(np.std(nrmse_numpy_list, ddof=0)) if len(nrmse_numpy_list) > 0 else float('nan')

    print(f"[{dataset_name} | {perm_method} icl={n_examples}] rmse: {avg_rmse:.6f} ± {std_rmse:.6f}")
    print(f"nrmse: {avg_nrmse:.6f} ± {std_nrmse:.6f} | steps_agent: {avg_steps_agent:.2f} ± {std_steps_agent:.2f}")
    print(f"steps_numpy: {avg_steps_numpy:.2f} ± {std_steps_numpy:.2f}")
    print(f"numpy_oracle_rmse: {avg_rmse_numpy:.6f} ± {std_rmse_numpy:.6f}")
    print(f"numpy_oracle_nrmse: {avg_nrmse_numpy:.6f} ± {std_nrmse_numpy:.6f}")

    metrics_path = os.path.join(dest_root, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"dataset {dataset_name}\n")
        f.write(f"perm_method {perm_method}\n")
        f.write(f"icl_examples {n_examples}\n")
        f.write(f"samples {len(idx_list)}\n\n")
        f.write(f"RMSE agent vs target {avg_rmse:.6f} ± {std_rmse:.6f}\n")
        f.write(f"NRMSE 0 to 1 {avg_nrmse:.6f} ± {std_nrmse:.6f}\n")
        f.write(f"Avg steps agent {avg_steps_agent:.2f} ± {std_steps_agent:.2f}\n")
        f.write(f"Avg steps numpy_oracle {avg_steps_numpy:.2f} ± {std_steps_numpy:.2f}\n")
        f.write(f"RMSE numpy_oracle vs target {avg_rmse_numpy:.6f} ± {std_rmse_numpy:.6f}\n")
        f.write(f"NRMSE 0 to 1 numpy_oracle {avg_nrmse_numpy:.6f} ± {std_nrmse_numpy:.6f}\n")

for n_examples in icl_examples_list:
    print(f"running icl={n_examples}")
    for m in perm_methods:
        run_deflation_for_method(m, indexes, k_blocks=k_blocks_default, n_examples=n_examples, dataset_name=dataset_name)