import json
import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_qwen_vl(model_id="Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_impl="flash_attention_2",
                 min_pixels=28*28*16, max_pixels=28*28*64):
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype, attn_implementation=attn_impl, device_map=device_map,)
    processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)

    return model, processor


def build_messages_for_similarity(img_path_a, img_path_b, ssim_value, ssim_threshold=0.65):
    sys_rules = (
        "You are an expert evaluation assistant. Judge whether the second image is a good rank 1 approximation of the first.\n"
        "Focus on overall structure, shapes.\n"
        "Respond with STRICT JSON only, no extra text, in the format:\n"
        '{ "similar": true|false, "justification": "<one short sentence>" }\n'
        "Do NOT output anything except valid JSON.\n"
    )
    user_q = "Compare Image A and Image B. Are they visually similar for our task?"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_rules}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path_a},
                {"type": "image", "image": img_path_b},
                {"type": "text", "text": user_q},
            ],
        },
    ]
    return messages

def build_messages_for_deflation(img_path_a, img_path_b, norm_value, norm_threshold=0.65):
    sys_rules = (
        "You are an expert evaluation assistant. Your task is to determine whether the second image still contains any meaningful structure, patterns, or shapes that resemble the first image.\n"
        "Even faint or partial structure counts as meaningful.\n"
        "If the second image appears to contain only noise or no trace of the original structure, respond with 'similar: false'.\n\n"
        "Focus on high-level features such as shape outlines or recognizable patterns, not small pixel differences.\n"
        "Respond with STRICT JSON only, no extra text, in the format:\n"
        '{ "similar": true|false, "justification": "<one short sentence>" }\n'
        "Do NOT output anything except valid JSON.\n"
    )
    user_q = "Compare Image A and Image B. Are they visually similar for our task?"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": sys_rules}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path_a},
                {"type": "image", "image": img_path_b},
                {"type": "text", "text": user_q},
            ],
        },
    ]
    return messages


def _extract_json(text):

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    snippet = m.group(0)
    try:
        return json.loads(snippet)
    except Exception as e:
        repaired = snippet.replace("True", "true").replace("False", "false").replace("'", '"')
        return json.loads(repaired)


def qwen_similarity_verdict(model, processor, img_path_a, img_path_b, ssim_value, ssim_threshold=0.65, max_new_tokens=128):

    messages = build_messages_for_similarity(img_path_a, img_path_b, ssim_value, ssim_threshold)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    print(text)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    max_input_length = 32768
    if inputs["input_ids"].shape[1] > max_input_length:
        inputs["input_ids"] = inputs["input_ids"][:, :max_input_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_length]

    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    try:
        data = _extract_json(output_text)
        similar = bool(data.get("similar", False))
        justification = str(data.get("justification", "")).strip()
    except Exception:
        similar = False
        justification = "Unable to parse model JSON."

    return {
        "similar": similar,
        "justification": justification,
        "raw_text": output_text,
    }

def qwen_deflation_verdict(model, processor, img_path_a, img_path_b, norm_value, norm_threshold=0.65, max_new_tokens=128):

    messages = build_messages_for_deflation(img_path_a, img_path_b, norm_value, norm_threshold)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    print(text)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    max_input_length = 32768
    if inputs["input_ids"].shape[1] > max_input_length:
        inputs["input_ids"] = inputs["input_ids"][:, :max_input_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_length]

    inputs = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    try:
        data = _extract_json(output_text)
        similar = bool(data.get("similar", False))
        justification = str(data.get("justification", "")).strip()
    except Exception:
        similar = False
        justification = "Unable to parse model JSON."

    return {
        "similar": similar,
        "justification": justification,
        "raw_text": output_text,
    }

def final_accept_reject(ssim_value, qwen_similar, ssim_threshold=0.5):
    if qwen_similar:
        return "accept", f"Accepted by evaluator (Qwen). SSIM={ssim_value:.4f} (threshold {ssim_threshold:.2f} for logging only)."
    else:
        return "reject", f"Rejected by evaluator (Qwen). SSIM={ssim_value:.4f}."

def evaluate_pair_with_qwen(model, processor, img_path_true, img_path_gemini, ssim_value, ssim_threshold=0.5):
    verdict = qwen_similarity_verdict(
        model, processor, img_path_true, img_path_gemini, ssim_value, ssim_threshold
    )
    decision, reason = final_accept_reject(ssim_value, verdict["similar"], ssim_threshold)
    return {
        "decision": decision,
        "reason": reason,
        "qwen_similar": verdict["similar"],
        "qwen_justification": verdict["justification"],
        "qwen_raw": verdict["raw_text"],
        "ssim": ssim_value,
        "threshold": ssim_threshold,
    }

def evaluate_deflation_with_qwen(model, processor, img_path_true, img_path_gemini, norm_value, norm_threshold=0.5):
    verdict = qwen_deflation_verdict(
        model, processor, img_path_true, img_path_gemini, norm_value, norm_threshold
    )
    decision, reason = final_accept_reject(norm_value, verdict["similar"], norm_threshold)
    return {
        "decision": decision,
        "reason": reason,
        "qwen_similar": verdict["similar"],
        "qwen_justification": verdict["justification"],
        "qwen_raw": verdict["raw_text"],
        "ssim": norm_value,
        "threshold": norm_threshold,
    }