# -*- coding: utf-8 -*-
"""
A tiny, dependency-free MVP for text -> FLAME-parameter editing.

This is intentionally conservative: it edits only existing motion FLAME params
(expression / jaw / eyes / head rotation / small shape deltas) and clamps values.
It is meant as a first landing point for research prototyping, not a trained model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


@dataclass(frozen=True)
class EditIntent:
    action: str
    strength: float


# Default semantic directions.  FLAME expression bases are not perfectly semantic,
# so these are deliberately exposed in one place for easy calibration after visual tests.
# Each tuple is: (parameter_key, dimension_index, coefficient).
DIRECTION_TABLE: Dict[str, Tuple[Tuple[str, int, float], ...]] = {
    "smile": (("expr", 0, 0.80), ("expr", 1, 0.35), ("expr", 2, -0.15)),
    "mouth_open": (("jaw_pose", 0, 0.28), ("expr", 3, 0.45)),
    "mouth_close": (("jaw_pose", 0, -0.22), ("expr", 3, -0.35)),
    "blink": (("eyes_pose", 1, 0.35), ("eyes_pose", 4, 0.35), ("expr", 8, 0.25)),
    "brow_up": (("expr", 4, 0.45), ("expr", 5, 0.25)),
    "frown": (("expr", 0, -0.55), ("expr", 6, 0.35), ("expr", 7, 0.25)),
    "look_left": (("eyes_pose", 0, 0.28), ("eyes_pose", 3, 0.28)),
    "look_right": (("eyes_pose", 0, -0.28), ("eyes_pose", 3, -0.28)),
    "look_up": (("eyes_pose", 1, -0.20), ("eyes_pose", 4, -0.20)),
    "look_down": (("eyes_pose", 1, 0.20), ("eyes_pose", 4, 0.20)),
    "turn_left": (("rotation", 1, 0.22),),
    "turn_right": (("rotation", 1, -0.22),),
    # Shape edits are intentionally tiny to avoid identity drift.
    "face_thin": (("betas", 0, -0.18), ("betas", 1, 0.08)),
    "chin_sharp": (("betas", 2, 0.12),),
    "nose_high": (("betas", 3, 0.10),),
}

KEYWORDS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("笑", "微笑", "开心", "smile", "happy"), "smile"),
    (("张嘴", "嘴巴张开", "开口", "open mouth", "mouth open"), "mouth_open"),
    (("闭嘴", "合嘴", "close mouth", "mouth close"), "mouth_close"),
    (("眨眼", "闭眼", "blink", "close eyes"), "blink"),
    (("抬眉", "挑眉", "眉毛上", "brow up", "raise eyebrow"), "brow_up"),
    (("皱眉", "生气", "不开心", "frown", "angry"), "frown"),
    (("看左", "向左看", "look left"), "look_left"),
    (("看右", "向右看", "look right"), "look_right"),
    (("看上", "向上看", "look up"), "look_up"),
    (("看下", "向下看", "look down"), "look_down"),
    (("头左", "左转头", "turn left"), "turn_left"),
    (("头右", "右转头", "turn right"), "turn_right"),
    (("脸瘦", "瘦脸", "thin face", "slimmer face"), "face_thin"),
    (("下巴尖", "尖下巴", "sharp chin"), "chin_sharp"),
    (("鼻梁高", "高鼻梁", "nose higher", "high nose"), "nose_high"),
)

INTENSITY_WORDS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("一点", "稍微", "轻微", "slightly", "a little"), 0.35),
    (("更", "明显", "more"), 0.60),
    (("很", "非常", "特别", "大幅", "夸张", "very", "strong", "extreme"), 0.90),
)

_CLAMP_TABLE = {
    "expr": (-3.0, 3.0),
    "jaw_pose": (-0.8, 0.8),
    "eyes_pose": (-0.8, 0.8),
    "rotation": (-0.8, 0.8),
    "neck_pose": (-0.8, 0.8),
    "translation": (-5.0, 5.0),
    "betas": (-2.0, 2.0),
}


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _parse_strength(text: str, default_strength: float) -> float:
    strength = default_strength
    for words, value in INTENSITY_WORDS:
        if any(w in text for w in words):
            strength = max(strength, value)
    return float(max(0.0, min(1.0, strength)))


def parse_text_edit(text: str, default_strength: float = 0.5) -> List[EditIntent]:
    """Parse Chinese/English edit text into conservative semantic edit intents."""
    text = _normalise_text(text)
    if not text:
        return []
    strength = _parse_strength(text, default_strength)
    intents: List[EditIntent] = []
    seen = set()
    for words, action in KEYWORDS:
        if action in seen:
            continue
        if any(w in text for w in words):
            intents.append(EditIntent(action=action, strength=strength))
            seen.add(action)
    return intents


def _apply_delta_to_tensor(tensor: torch.Tensor, index: int, delta: float) -> torch.Tensor:
    if tensor is None or tensor.numel() == 0 or tensor.shape[-1] <= index:
        return tensor
    tensor = tensor.clone()
    tensor[..., index] = tensor[..., index] + tensor.new_tensor(delta)
    return tensor


def _clamp_flame_params(flame_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key, (lo, hi) in _CLAMP_TABLE.items():
        if key in flame_params and torch.is_tensor(flame_params[key]):
            flame_params[key] = torch.clamp(flame_params[key], lo, hi)
    return flame_params


def apply_text_edit(
    flame_params: Dict[str, torch.Tensor],
    text: str,
    strength: float = 0.5,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Apply text-command edits to a batched FLAME parameter dict.

    The input dict is copied shallowly and tensors are cloned only when edited,
    so the caller can safely keep the original motion params if needed.
    """
    edited = dict(flame_params)
    intents = parse_text_edit(text, default_strength=strength)
    if verbose:
        print("[TextEdit] prompt=", repr(text), "intents=", intents)
        print("[TextEdit] available FLAME keys=", {k: tuple(v.shape) for k, v in edited.items() if torch.is_tensor(v)})
    for intent in intents:
        for key, index, coeff in DIRECTION_TABLE.get(intent.action, ()):
            if key not in edited or not torch.is_tensor(edited[key]):
                continue
            delta = coeff * intent.strength
            edited[key] = _apply_delta_to_tensor(edited[key], index, delta)
            if verbose:
                print(f"[TextEdit] {intent.action}: {key}[..., {index}] += {delta:.4f}")
    return _clamp_flame_params(edited)
