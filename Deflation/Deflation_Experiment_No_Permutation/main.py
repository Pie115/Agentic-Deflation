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

def save_deflation_trajectory(unpermuted_steps, path="deflation_trajectory.png"):
    assert len(unpermuted_steps) >= 1
    steps = len(unpermuted_steps)

    fig, axes = plt.subplots(1, steps, figsize=(3 * steps, 3))
    if steps == 1:
        axes = np.array([axes])

    for j in range(steps):
        ax = axes[j]
        ax.imshow(unpermuted_steps[j], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"Step {j}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")

index = 0
k = 1

number_of_digits = 50
indexes = random.sample(range(1, 1797), number_of_digits)

genai.configure(api_key="AIzaSyDRlQ0OvsZLjnZS28jMxEwxriPMHOrLHBQ")
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

qwen_model, processor = load_qwen_vl()
rmse_total = 0
nrmse_total = 0

while (index < len(indexes)):
    img, number = get_digits_image(indexes[index])
    img_current = img.copy()  # residual we update
    save_single_image(img, "original_no_deflation.png")

    steps_images = [img.copy()]
    deflation_decision = 'accept'

    while (deflation_decision == 'accept'):
        # Build ICL prompt on the CURRENT residual (unpermuted)
        prompt, ex1_matrix, ex2_matrix = build_solver_prompt(
            img_current, number, k=k
        )
        resp = gemini_model.generate_content([prompt])
        print(prompt)
        print("========================Solver Response=========================")
        print("Gemini output:\n", resp.text)

        gemini_decision = 'reject'
        max_gemini_retries = 8
        gemini_attempts = 0

        while (gemini_decision == 'reject'):
            try:
                u_pred, s_pred, v_pred = parse_gemini_response(resp.text)
                A_gemini = reconstruct_rankk_from_uvs(u_pred, s_pred, v_pred)
            except Exception as e:
                print("Parse/reconstruct error:", e)
                A_gemini = None  # force retry

            # If reconstruction failed or shape mismatch, retry Gemini
            if A_gemini is None or A_gemini.shape != img_current.shape:
                print("Shape mismatch or invalid reconstruction; retrying Gemini...")
                gemini_attempts += 1
                if gemini_attempts > max_gemini_retries:
                    print("Max Gemini retries reached; skipping this digit.")
                    deflation_decision = 'reject'
                    index += 1
                    break
                resp = gemini_model.generate_content([prompt])
                continue

            # Metrics vs ground-truth SVD (optional monitoring)
            A_true = true_rankk(img_current, k=k)

            try:
                img_f = img_current.astype(float)
                true_f = A_true.astype(float)
                gem_f = A_gemini.astype(float)

                ssim_true = ssim(img_f, true_f, data_range=255)
                ssim_gemini = ssim(img_f, gem_f, data_range=255)
                rmse_gemini_vs_true = np.sqrt(np.mean((gem_f - true_f) ** 2))
            except ValueError as e:
                print(f"SSIM computation error: {e}")
                gemini_attempts += 1
                if gemini_attempts > max_gemini_retries:
                    print("Max Gemini retries reached after SSIM errors; skipping this digit.")
                    deflation_decision = 'reject'
                    index += 1
                    break
                resp = gemini_model.generate_content([prompt])
                continue

            print("========================SSIM / RMSE (current residual)=========================")
            print(f"SSIM (True SVD vs Current Residual): {ssim_true:.4f}")
            print(f"SSIM (Gemini vs Current Residual): {ssim_gemini:.4f}")
            print(f"RMSE (Gemini vs True SVD): {rmse_gemini_vs_true:.4f}")
            nrmse_255 = rmse_gemini_vs_true / 255.0
            print(f"NRMSE (÷255): {nrmse_255:.6f}")

            # Qwen rank-1 evaluator: compare ORIGINAL vs CANDIDATE rank-1 on CURRENT residual
            save_single_image(img_current, "current_residual.png")
            save_single_image(A_gemini, "gemini_rank1.png")
            result = evaluate_pair_with_qwen(
                qwen_model, processor,
                img_path_true="current_residual.png",
                img_path_gemini="gemini_rank1.png"
            )

            print("========================SVD Evaluator Response=========================")
            if (result["decision"] == 'reject'):
                if os.path.exists('gemini_rank1.png'):
                    os.remove('gemini_rank1.png')
                resp = gemini_model.generate_content([prompt])
                gemini_attempts += 1
                if gemini_attempts > max_gemini_retries:
                    print("Max Gemini retries reached after evaluator rejection; skipping this digit.")
                    deflation_decision = 'reject'
                    index += 1
                    break
            else:
                gemini_decision = 'accept'
                if os.path.exists('gemini_rank1.png'):
                    os.remove('gemini_rank1.png')
                rmse_total += rmse_gemini_vs_true
                nrmse_total += nrmse_255

        if deflation_decision == 'reject':
            # Either finished or bailed due to too many retries; save what we have
            dest_folder = "final_results_deflation_unperm"
            os.makedirs(dest_folder, exist_ok=True)
            traj_path = os.path.join(dest_folder, f"deflation_trajectory_{indexes[index-1] if index>0 else indexes[index]}.png")
            save_deflation_trajectory(steps_images, path=traj_path)
            continue

        # Subtract accepted rank-1, clip at 0 (residual update)
        img_current = np.clip(img_current - A_gemini, 0, None).astype(int)
        save_single_image(img_current, "deflated.png")

        # Track trajectory (unpermuted only)
        steps_images.append(img_current.copy())

        # Qwen deflation evaluator: **ORIGINAL vs CURRENT RESIDUAL**
        deflation_result = evaluate_deflation_with_qwen(
            qwen_model, processor,
            img_path_true="original_no_deflation.png",
            img_path_gemini="deflated.png"
        )

        print("========================Deflation Evaluator Response=========================")
        print(deflation_result)

        if (deflation_result["decision"] == 'reject'):
            deflation_decision = 'reject'

            dest_folder = "final_results_deflation_unperm"
            os.makedirs(dest_folder, exist_ok=True)
            traj_path = os.path.join(dest_folder, f"deflation_trajectory_{indexes[index]}.png")
            save_deflation_trajectory(steps_images, path=traj_path)

            index += 1

print("rmse:", rmse_total/50, "nrmse:", nrmse_total/50)
