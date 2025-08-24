import numpy as np
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from skimage.metrics import structural_similarity as ssim
import random
from prompt_qwen import load_qwen_vl, evaluate_pair_with_qwen
from prompt_gemini import (
    get_digits_image, compute_svd_lists, build_solver_prompt, parse_gemini_response,
    reconstruct_rankk_from_uvs, true_rankk, save_single_image,
    save_comparison_with_examples
)
import shutil

index = 0
k = 1

number_of_digits = 50
indexes = random.sample(range(1, 1797), number_of_digits)

genai.configure(api_key="AIzaSyDRlQ0OvsZLjnZS28jMxEwxriPMHOrLHBQ")
gemini_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

qwen_model, processor = load_qwen_vl()
rmse_total = 0
nrmse_total = 0

while index < len(indexes):
    img, number = get_digits_image(indexes[index])

    # ---- NO PERMUTATION: build prompt on the raw matrix ----
    prompt, ex1_matrix, ex2_matrix = build_solver_prompt(img, number, k=k)
    resp = gemini_model.generate_content([prompt])
    print(prompt)
    print("========================Solver Response=========================")
    print("Gemini output:\n", resp.text)

    try:
        u_pred, s_pred, v_pred = parse_gemini_response(resp.text)
        A_gemini = reconstruct_rankk_from_uvs(u_pred, s_pred, v_pred)
    except Exception as e:
        print("Parse/reconstruct error:", e)
        A_gemini = np.zeros_like(img)

    A_true = true_rankk(img, k=k)

    img_f = img.astype(float)
    true_f = A_true.astype(float)
    gem_f = A_gemini.astype(float)

    try:
        ssim_true = ssim(img_f, true_f, data_range=255)
        ssim_gemini = ssim(img_f, gem_f, data_range=255)
        ssim_gemini_vs_true = ssim(gem_f, true_f, data_range=255)
        rmse_gemini_vs_true = np.sqrt(np.mean((gem_f - true_f) ** 2))
    except ValueError as e:
        print(f"SSIM computation error: {e}")
        ssim_true = 0.0
        ssim_gemini = 0.0
        ssim_gemini_vs_true = 0.0
        result = {"decision": "reject", "reason": "Dimension mismatch during SSIM."}
        continue

    print("========================SSIM Values=========================")
    print(f"SSIM (True SVD vs Original): {ssim_true:.4f}")
    print(f"SSIM (Gemini vs Original): {ssim_gemini:.4f}")
    print(f"SSIM (Gemini vs True SVD): {ssim_gemini_vs_true:.4f}")
    print(f"RMSE (Gemini vs True SVD): {rmse_gemini_vs_true:.4f}")
    nrmse_255 = rmse_gemini_vs_true / 255.0
    print(f"NRMSE (÷255): {nrmse_255:.6f}")

    # Save images for the evaluator (raw, no blocky)
    save_single_image(img, "original.png")
    save_single_image(A_gemini, "gemini_rankk_only.png")

    # Qwen evaluator: no SSIM inputs anymore
    result = evaluate_pair_with_qwen(
        qwen_model, processor,
        img_path_true="original.png",
        img_path_gemini="gemini_rankk_only.png",
    )

    print("========================Evaluator Response=========================")
    if result["decision"] == 'reject':
        if os.path.exists('gemini_rankk_only.png'):
            os.remove('gemini_rankk_only.png')
    else:
        # Single-row comparison figure (examples, original, gemini, true)
        save_comparison_with_examples(
            ex1_matrix, ex2_matrix, img, A_gemini, A_true,
            path="digits_rankk_comparison.png"
        )

        if os.path.exists('gemini_rankk_only.png'):
            os.remove('gemini_rankk_only.png')

        dest_folder = "final_results_no_perm"
        os.makedirs(dest_folder, exist_ok=True)
        file_base = f"digits_rankk_comparison_{indexes[index]}"
        shutil.move("digits_rankk_comparison.png",
                    os.path.join(dest_folder, file_base + ".png"))
        index += 1
        rmse_total += rmse_gemini_vs_true
        nrmse_total += nrmse_255

    print(result)

print("rmse:", rmse_total/50, "nrmse:", nrmse_total/50)
