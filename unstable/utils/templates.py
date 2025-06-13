import re
from typing import Tuple, Dict


def apply_default_template(observation: str) -> str:
    return (
        "You are playing a two-player zero-sum game. Make valid moves to win. "
        "You should first reason about your next move, and then submit the move enclosed by \\boxed{}."
        f"Observation: {observation}\n"
    )

def qwen3_template(observation: str) -> str:
    return (
        f"<|im_start|>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def qwen3_template_reasoning(observation: str) -> str:
    return (
        "<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n"
        f"Question: {observation}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def gemma3_template(observation: str) -> str:
    return (
        f"<bos><start_of_turn>user\nYou are playing a two-player zero-sum game. Make valid actions to win.\nObservation: {observation}"
        "\nPlease reason step by step, and put your final answer within \\boxed{}.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def extract_action_and_format_feedback(raw_action: str) -> Tuple[str, Dict[str, bool]]:
    matches = re.findall(r"\\boxed\{(.*?)\}", raw_action)
    
    if matches:
        last_match = matches[-1].strip()
        if last_match:  # non-empty boxed
            action = f"[{last_match}]" if "[" not in last_match else last_match
            has_think = 1
        else:  # empty boxed
            action = raw_action
            has_think = 0
    else:  # no boxed at all
        action = raw_action
        has_think = 0

    format_feedback = {"has_think": bool(has_think), "has_answer": False, "order_correct": False}
    return action, format_feedback


OBSERVATION_FORMATTING = {
    "default": apply_default_template,
    "qwen3": qwen3_template,
    "qwen3-game": qwen3_template,
    "qwen3-zs": qwen3_template,
    "qwen3-reasoning": qwen3_template_reasoning,
    "gemma3-zs": gemma3_template
}

ACTION_EXTRACTION = {
    "default": extract_action_and_format_feedback,
    "qwen3-zs": extract_action_and_format_feedback,
}
