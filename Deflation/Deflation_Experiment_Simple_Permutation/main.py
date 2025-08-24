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

index = 0
k = 1

number_of_digits = 50
indexes = random.sample(range(1, 1797), number_of_digits)

genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

qwen_model, processor = load_qwen_vl()
rmse_total = 0
nrmse_total = 0

while index < len(indexes):
    img, number = get_digits_image(indexes[index])
    og_img = img  # original (unpermuted)

    # ==== Simple (naive) permutation computed ONCE per image ====
    row_order = np.argsort(-img.sum(axis=1))
    col_order = np.argsort(-img.sum(axis=0))
    row_inv = np.argsort(row_order)
    col_inv = np.argsort(col_order)
    img_perm = img[np.ix_(row_order, col_order)]

    # Save the original permuted image for residual-vs-original comparisons
    save_single_image(img_perm, "original_blocky_no_deflation.png")

    permuted_steps = [img_perm.copy()]
    unpermuted_steps = [img.copy()]

    deflation_decision = 'accept'

    while deflation_decision == 'accept':
        # ICL prompt uses the SAME permutation for both examples and target
        prompt, ex1_matrix, ex2_matrix = build_solver_prompt(
            img_perm, number, k=k, row_order=row_order, col_order=col_order
        )
        resp = gemini_model.generate_content([prompt])
        print(prompt)
        print("========================Solver Response=========================")
        print("Gemini output:\n", resp.text)

        gemini_decision = 'reject'
        max_gemini_retries = 8
        gemini_attempts = 0

        while gemini_decision == 'reject':
            try:
                u_pred, s_pred, v_pred = parse_gemini_response(resp.text)
                A_gemini_perm = reconstruct_rankk_from_uvs(u_pred, s_pred, v_pred)
            except Exception as e:
                print("Parse/reconstruct error:", e)
                A_gemini_perm = None  # force retry

            # Retry if reconstruction failed or wrong shape
            if A_gemini_perm is None or A_gemini_perm.shape != img_perm.shape:
                print("Shape mismatch or invalid reconstruction; retrying Gemini...")
                gemini_attempts += 1
                if gemini_attempts > max_gemini_retries:
                    print("Max Gemini retries reached; skipping this digit.")
                    deflation_decision = 'reject'
                    index += 1
                    break
                resp = gemini_model.generate_content([prompt])
                continue

            # Metrics (for logging only)
            A_true = true_rankk(img, k=k)
            A_true_blocky = true_rankk(img_perm, k=k)

            img_f = img.astype(float)
            true_f = A_true.astype(float)
            img_perm_f = img_perm.astype(float)
            true_blocky_f = A_true_blocky.astype(float)
            gem_perm_f = A_gemini_perm.astype(float)

            try:
                ssim_true = ssim(img_f, true_f, data_range=255)
                ssim_gemini = ssim(img_perm_f, gem_perm_f, data_range=255)
                ssim_gemini_vs_true = ssim(gem_perm_f, true_blocky_f, data_range=255)
                rmse_gemini_vs_true = np.sqrt(np.mean((gem_perm_f - true_blocky_f) ** 2))
            except ValueError as e:
                print(f"SSIM computation error: {e}")
                # auto-retry Gemini on SSIM issues
                gemini_attempts += 1
                if gemini_attempts > max_gemini_retries:
                    print("Max Gemini retries reached after SSIM errors; skipping this digit.")
                    deflation_decision = 'reject'
                    index += 1
                    break
                resp = gemini_model.generate_content([prompt])
                continue

            print("========================SSIM Values=========================")
            print(f"SSIM (True SVD vs Original): {ssim_true:.4f}")
            print(f"SSIM (Gemini(blocky) vs Original(blocky)): {ssim_gemini:.4f}")
            print(f"SSIM (Gemini(blocky) vs True(blocky)): {ssim_gemini_vs_true:.4f}")
            print(f"RMSE (Gemini vs True SVD): {rmse_gemini_vs_true:.4f}")
            nrmse_255 = rmse_gemini_vs_true / 255.0
            print(f"NRMSE (÷255): {nrmse_255:.6f}")

            # Qwen rank-1 evaluator (permuted original vs permuted candidate)
            save_single_image(img_perm, "original_blocky.png")
            save_single_image(A_gemini_perm, "gemini_rankk_only.png")
            result = evaluate_pair_with_qwen(
                qwen_model, processor,
                img_path_true="original_blocky.png",
                img_path_gemini="gemini_rankk_only.png"
            )

            print("========================SVD Evaluator Response=========================")
            if result["decision"] == 'reject':
                # cleanup and retry Gemini
                if os.path.exists('true_rankk_only.png'):
                    os.remove('true_rankk_only.png')
                if os.path.exists('gemini_rankk_only.png'):
                    os.remove('gemini_rankk_only.png')
                resp = gemini_model.generate_content([prompt])
                gemini_attempts += 1
                if gemini_attempts > max_gemini_retries:
                    print("Max Gemini retries reached after evaluator rejection; skipping this digit.")
                    deflation_decision = 'reject'
                    index += 1
                    break
            else:
                gemini_decision = 'accept'
                # cleanup
                if os.path.exists('true_rankk_only.png'):
                    os.remove('true_rankk_only.png')
                if os.path.exists('gemini_rankk_only.png'):
                    os.remove('gemini_rankk_only.png')
                rmse_total += rmse_gemini_vs_true
                nrmse_total += nrmse_255

        if deflation_decision == 'reject':
            # Finished/bailed for this digit: save trajectory and continue
            dest_folder = "final_results4"
            os.makedirs(dest_folder, exist_ok=True)
            traj_path = os.path.join(
                dest_folder,
                f"deflation_trajectory_{indexes[index-1] if index>0 else indexes[index]}.png"
            )
            save_deflation_trajectory(unpermuted_steps, permuted_steps, path=traj_path)
            continue

        # === Deflation step: subtract accepted rank-1, clip at 0 ===
        img_perm = np.clip(img_perm - A_gemini_perm, 0, None)
        save_single_image(img_perm, "deflated_blocky.png")

        # Track trajectory in both spaces
        unpermuted_residual = img_perm[np.ix_(row_inv, col_inv)]
        permuted_steps.append(img_perm.copy())
        unpermuted_steps.append(unpermuted_residual.copy())

        # === Deflation evaluator: residual-vs-original (both permuted) ===
        deflation_result = evaluate_deflation_with_qwen(
            qwen_model, processor,
            img_path_true="original_blocky_no_deflation.png",  # fixed original (permuted)
            img_path_gemini="deflated_blocky.png"              # current residual (permuted)
        )

        print("========================Deflation Evaluator Response=========================")
        print(deflation_result)

        if deflation_result["decision"] == 'reject':
            deflation_decision = 'reject'

            # Save trajectory figure
            dest_folder = "final_results4"
            os.makedirs(dest_folder, exist_ok=True)
            traj_path = os.path.join(dest_folder, f"deflation_trajectory_{indexes[index]}.png")
            save_deflation_trajectory(unpermuted_steps, permuted_steps, path=traj_path)

            index += 1

print("rmse:", rmse_total / 50, "nrmse:", nrmse_total / 50)