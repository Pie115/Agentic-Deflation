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
    save_dual_comparison_with_examples
)
import shutil
from permute_utils import groupnteach_permutation

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

    # k_blocks=2 by default for 8x8
    k_blocks = 2
    img_perm_float, row_order, col_order, row_inv, col_inv, groups, D = groupnteach_permutation(
        img.astype(float), k=k_blocks, density_mode="mean", random_state=0
    )
    img_perm = img_perm_float.astype(int)

    prompt, ex1_matrix, ex2_matrix = build_solver_prompt(
        img_perm, number, k=k, row_order=row_order, col_order=col_order
    )
    resp = gemini_model.generate_content([prompt])
    print(prompt)
    print("========================Solver Response=========================")
    print("Gemini output:\n", resp.text)

    try:
        u_pred, s_pred, v_pred = parse_gemini_response(resp.text)
        A_gemini_perm = reconstruct_rankk_from_uvs(u_pred, s_pred, v_pred)
    except Exception as e:
        print("Parse/reconstruct error:", e)
        A_gemini_perm = np.zeros_like(img_perm)

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
        ssim_true = 0.0
        ssim_gemini = 0.0
        ssim_gemini_vs_true = 0.0
        result = {"decision": "reject", "reason": "Dimension mismatch during SSIM."}
        continue

    print("========================SSIM Values=========================")
    print(f"SSIM (True SVD vs Original): {ssim_true:.4f}")
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
        ssim_value=ssim_gemini,                                
        ssim_threshold=0.65
    )

    print("========================Evaluator Response=========================")
    if (result["decision"] == 'reject'):
        if os.path.exists('true_rankk_only.png'):
            os.remove('true_rankk_only.png')
        if os.path.exists('gemini_rankk_only.png'):
            os.remove('gemini_rankk_only.png')
    else:
        # CHANGED: row_inv/col_inv already computed by groupnteach_permutation
        A_gemini = A_gemini_perm[np.ix_(row_inv, col_inv)]

        ex1_blocky = ex1_matrix[np.ix_(row_order, col_order)]
        ex2_blocky = ex2_matrix[np.ix_(row_order, col_order)]

        save_dual_comparison_with_examples(
            ex1_matrix, ex2_matrix, img, A_gemini, A_true,
            ex1_blocky, ex2_blocky, img_perm, A_gemini_perm, A_true_blocky,
            path="digits_rankk_comparison_dual.png"
        )

        if os.path.exists('true_rankk_only.png'):
            os.remove('true_rankk_only.png')
        if os.path.exists('gemini_rankk_only.png'):
            os.remove('gemini_rankk_only.png')

        dest_folder = "final_results3"
        os.makedirs(dest_folder, exist_ok=True)
        file_base = f"digits_rankk_comparison_{indexes[index]}"
        shutil.move("digits_rankk_comparison_dual.png", os.path.join(dest_folder, file_base + "_dual.png"))
        index += 1    
        rmse_total += rmse_gemini_vs_true
        nrmse_total += nrmse_255

    print(result)
print("rmse:", rmse_total/50, "nrmse:", nrmse_total/50)