import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from skimage.metrics import structural_similarity as ssim
import random
from prompt_qwen import load_qwen_vl, evaluate_pair_with_qwen
from prompt_gemini import get_digits_image, compute_svd_lists, build_solver_prompt, parse_gemini_response, reconstruct_rankk_from_uvs, true_rankk, save_single_image, save_dual_comparison_variable_examples, save_comparison_variable_examples
import shutil
from permute_utils import groupnteach_permutation
import time

k = 1
n_per_method = 5
perm_methods = ['none', 'sort', 'group_n_teach']
icl_examples_list = [0, 1, 2, 3, 4]
k_blocks_default = 2

genai.configure(api_key="AIzaSyCW_JDKAVHDpN9s_aHBftBmcAkxJoDCSjg")
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")
qwen_model, processor = load_qwen_vl()

all_indices = random.sample(range(1, 1797), n_per_method)

def run_svd_only_for_method(perm_method, indexes, k=1, k_blocks=k_blocks_default, n_examples=2):
    gem_vs_orig_rmse = []
    gem_vs_orig_ssim = []
    gem_vs_orig_nrmse = []

    svd_vs_orig_rmse = []
    svd_vs_orig_ssim = []
    svd_vs_orig_nrmse = []

    gem_vs_svd_rmse = []
    gem_vs_svd_ssim = []
    gem_vs_svd_nrmse = []

    perm_rmse_monitor = []
    perm_nrmse_monitor = []

    total_retries = 0
    retries_per_sample = []

    index = 0
    while index < len(indexes):
        img, number = get_digits_image(indexes[index])

        if perm_method == 'group_n_teach':
            img_perm_float, row_order, col_order, row_inv, col_inv, groups, D = groupnteach_permutation(
                img.astype(float), k=k_blocks, density_mode="mean", random_state=0
            )
            img_perm = img_perm_float.astype(int)
            prompt, ex_list_unperm, ex_list_perm = build_solver_prompt(
                img_perm, number, k=k, perm_method=perm_method, row_order=row_order, col_order=col_order, n_examples=n_examples
            )
        elif perm_method == 'sort':
            row_order = np.argsort(-img.sum(axis=1))
            col_order = np.argsort(-img.sum(axis=0))
            row_inv = np.argsort(row_order)
            col_inv = np.argsort(col_order)
            img_perm = img[np.ix_(row_order, col_order)]
            prompt, ex_list_unperm, ex_list_perm = build_solver_prompt(
                img_perm, number, k=k, perm_method=perm_method, row_order=row_order, col_order=col_order, n_examples=n_examples
            )
        else:
            img_perm = img
            prompt, ex_list_unperm, ex_list_perm = build_solver_prompt(
                img, number, k=k, perm_method=perm_method, n_examples=n_examples
            )

        retries_this_sample = 0
        while True:
            try:
                resp = gemini_model.generate_content([prompt])
            except Exception as e:
                print(f"API error {type(e).__name__}. sleeping 10s")
                time.sleep(10)
                continue

            try:
                u_pred, s_pred, v_pred = parse_gemini_response(resp.text)
                A_gemini_perm = reconstruct_rankk_from_uvs(u_pred, s_pred, v_pred)
                if A_gemini_perm.shape != img_perm.shape:
                    raise ValueError("shape mismatch")
            except Exception as e:
                print(f"parse or reconstruct error {e}")
                retries_this_sample += 1
                total_retries += 1
                continue

            A_true = true_rankk(img, k=k)
            A_true_blocky = true_rankk(img_perm, k=k)

            if perm_method == 'none':
                A_gemini_orig = A_gemini_perm
                img_orig = img
                A_true_orig = A_true
            else:
                A_gemini_orig = A_gemini_perm[np.ix_(row_inv, col_inv)]
                img_orig = img
                A_true_orig = A_true

            try:
                ssim_gem_vs_orig = ssim(img_orig.astype(float), A_gemini_orig.astype(float), data_range=255)
                rmse_gem_vs_orig = np.sqrt(np.mean((A_gemini_orig.astype(float) - img_orig.astype(float))**2))
                nrmse_gem_vs_orig = rmse_gem_vs_orig / 255.0

                ssim_svd_vs_orig = ssim(img_orig.astype(float), A_true_orig.astype(float), data_range=255)
                rmse_svd_vs_orig = np.sqrt(np.mean((A_true_orig.astype(float) - img_orig.astype(float))**2))
                nrmse_svd_vs_orig = rmse_svd_vs_orig / 255.0

                ssim_gem_vs_svd = ssim(A_gemini_orig.astype(float), A_true_orig.astype(float), data_range=255)
                rmse_gem_vs_svd = np.sqrt(np.mean((A_gemini_orig.astype(float) - A_true_orig.astype(float))**2))
                nrmse_gem_vs_svd = rmse_gem_vs_svd / 255.0

                ssim_perm = ssim(img_perm.astype(float), A_gemini_perm.astype(float), data_range=255)
                rmse_perm = np.sqrt(np.mean((A_gemini_perm.astype(float) - A_true_blocky.astype(float))**2))
                nrmse_perm = rmse_perm / 255.0
            except ValueError:
                retries_this_sample += 1
                total_retries += 1
                continue

            save_single_image(img_perm, "original_blocky.png")
            save_single_image(A_gemini_perm, "gemini_rankk_only.png")
            result = evaluate_pair_with_qwen(
                qwen_model, processor,
                img_path_true="original_blocky.png",
                img_path_gemini="gemini_rankk_only.png",
                ssim_value=ssim_perm,
                ssim_threshold=0.65
            )
            if os.path.exists('gemini_rankk_only.png'):
                os.remove('gemini_rankk_only.png')
            if os.path.exists('original_blocky.png'):
                os.remove('original_blocky.png')

            if result["decision"] == 'reject':
                retries_this_sample += 1
                total_retries += 1
                continue
            else:
                gem_vs_orig_rmse.append(rmse_gem_vs_orig)
                gem_vs_orig_ssim.append(ssim_gem_vs_orig)
                gem_vs_orig_nrmse.append(nrmse_gem_vs_orig)

                svd_vs_orig_rmse.append(rmse_svd_vs_orig)
                svd_vs_orig_ssim.append(ssim_svd_vs_orig)
                svd_vs_orig_nrmse.append(nrmse_svd_vs_orig)

                gem_vs_svd_rmse.append(rmse_gem_vs_svd)
                gem_vs_svd_ssim.append(ssim_gem_vs_svd)
                gem_vs_svd_nrmse.append(nrmse_gem_vs_svd)

                perm_rmse_monitor.append(rmse_perm)
                perm_nrmse_monitor.append(nrmse_perm)
                break

        retries_per_sample.append(retries_this_sample)

        if perm_method != 'none':
            save_dual_comparison_variable_examples(
                ex_list_unperm, ex_list_perm,
                img, A_gemini_orig, A_true,
                img_perm, A_gemini_perm, A_true_blocky,
                path="digits_rankk_comparison_dual.png"
            )
        else:
            save_comparison_variable_examples(
                ex_list_unperm,
                img, A_gemini_perm, A_true,
                path="digits_rankk_comparison.png"
            )

        dest_folder = f"final_results_{perm_method}_icl{n_examples}"
        os.makedirs(dest_folder, exist_ok=True)
        file_base = f"digits_rankk_comparison_{indexes[index]}"
        if perm_method != 'none':
            shutil.move("digits_rankk_comparison_dual.png", os.path.join(dest_folder, file_base + "_dual.png"))
        else:
            shutil.move("digits_rankk_comparison.png", os.path.join(dest_folder, file_base + "_comparison.png"))

        index += 1

    def mean_std(arr):
        arr = np.array(arr, dtype=float)
        if arr.size == 0:
            return float('nan'), float('nan')
        return float(arr.mean()), float(arr.std(ddof=0))

    n = len(indexes)
    dest_folder = f"final_results_{perm_method}_icl{n_examples}"
    os.makedirs(dest_folder, exist_ok=True)
    metrics_path = os.path.join(dest_folder, "metrics.txt")

    g_o_rmse_m, g_o_rmse_s = mean_std(gem_vs_orig_rmse)
    g_o_nrmse_m, g_o_nrmse_s = mean_std(gem_vs_orig_nrmse)
    g_o_ssim_m, g_o_ssim_s = mean_std(gem_vs_orig_ssim)

    s_o_rmse_m, s_o_rmse_s = mean_std(svd_vs_orig_rmse)
    s_o_nrmse_m, s_o_nrmse_s = mean_std(svd_vs_orig_nrmse)
    s_o_ssim_m, s_o_ssim_s = mean_std(svd_vs_orig_ssim)

    g_s_rmse_m, g_s_rmse_s = mean_std(gem_vs_svd_rmse)
    g_s_nrmse_m, g_s_nrmse_s = mean_std(gem_vs_svd_nrmse)
    g_s_ssim_m, g_s_ssim_s = mean_std(gem_vs_svd_ssim)

    perm_rmse_m, perm_rmse_s = mean_std(perm_rmse_monitor)
    perm_nrmse_m, perm_nrmse_s = mean_std(perm_nrmse_monitor)

    retries_avg = np.mean(retries_per_sample) if len(retries_per_sample) > 0 else float('nan')
    retries_std = np.std(retries_per_sample, ddof=0) if len(retries_per_sample) > 0 else float('nan')

    with open(metrics_path, "w") as f:
        f.write(f"Permutation method {perm_method}\n")
        f.write(f"In context examples {n_examples}\n")
        f.write(f"Samples {n}\n")
        f.write(f"Total retries beyond first {total_retries}\n")
        f.write(f"Retries per sample {retries_avg:.4f} ± {retries_std:.4f}\n\n")

        f.write("Averages vs Original\n")
        f.write(f"RMSE Gemini vs Original {g_o_rmse_m:.6f} ± {g_o_rmse_s:.6f}\n")
        f.write(f"NRMSE 0 to 1 Gemini vs Original {g_o_nrmse_m:.6f} ± {g_o_nrmse_s:.6f}\n")
        f.write(f"SSIM Gemini vs Original {g_o_ssim_m:.6f} ± {g_o_ssim_s:.6f}\n")
        f.write(f"RMSE SVD vs Original {s_o_rmse_m:.6f} ± {s_o_rmse_s:.6f}\n")
        f.write(f"NRMSE 0 to 1 SVD vs Original {s_o_nrmse_m:.6f} ± {s_o_nrmse_s:.6f}\n")
        f.write(f"SSIM SVD vs Original {s_o_ssim_m:.6f} ± {s_o_ssim_s:.6f}\n\n")

        f.write("Averages Gemini vs SVD\n")
        f.write(f"RMSE Gemini vs SVD {g_s_rmse_m:.6f} ± {g_s_rmse_s:.6f}\n")
        f.write(f"NRMSE 0 to 1 Gemini vs SVD {g_s_nrmse_m:.6f} ± {g_s_nrmse_s:.6f}\n")
        f.write(f"SSIM Gemini vs SVD {g_s_ssim_m:.6f} ± {g_s_ssim_s:.6f}\n\n")

        f.write("Permuted space monitor reference only\n")
        f.write(f"RMSE Gemini vs True permuted {perm_rmse_m:.6f} ± {perm_rmse_s:.6f}\n")
        f.write(f"NRMSE 0 to 1 {perm_nrmse_m:.6f} ± {perm_nrmse_s:.6f}\n")

    print(f"[{perm_method} icl={n_examples}] wrote metrics -> {metrics_path}")

for n_examples in icl_examples_list:
    print(f"running icl={n_examples}")
    for m in perm_methods:
        print(f"method {m}")
        run_svd_only_for_method(m, all_indices, k=k, k_blocks=k_blocks_default, n_examples=n_examples)
