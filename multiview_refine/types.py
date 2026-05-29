from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch


TensorDict = Dict[str, torch.Tensor]


FLAME_KEYS = {
    "expr",
    "rotation",
    "neck_pose",
    "jaw_pose",
    "eyes_pose",
    "translation",
    "betas",
    "teeth_bs",
}


@dataclass
class MultiViewFrame:
    frame_id: str
    image_path: Path
    mask_path: Optional[Path]
    flame_param_path: Optional[Path]
    landmark_path: Optional[Path]
    c2w: torch.Tensor
    intr: torch.Tensor
    flame_params: TensorDict = field(default_factory=dict)
    landmarks_2d: Optional[torch.Tensor] = None


@dataclass
class MultiViewBatch:
    images: torch.Tensor
    masks: torch.Tensor
    c2ws: torch.Tensor
    intrs: torch.Tensor
    bg_colors: torch.Tensor
    flame_params: TensorDict
    frame_ids: List[str]
    landmarks_2d: Optional[torch.Tensor] = None
    view_indices: Optional[torch.Tensor] = None

    def index(self, indices: torch.Tensor | List[int]) -> "MultiViewBatch":
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        idx = indices.to(self.images.device)
        flame_params = {}
        for key, value in self.flame_params.items():
            if key == "betas":
                flame_params[key] = value
            else:
                flame_params[key] = value[:, idx]
        landmarks = None
        if self.landmarks_2d is not None:
            landmarks = self.landmarks_2d[:, idx]
        view_indices = idx
        if self.view_indices is not None:
            view_indices = self.view_indices.to(self.images.device)[idx]
        return MultiViewBatch(
            images=self.images[:, idx],
            masks=self.masks[:, idx],
            c2ws=self.c2ws[:, idx],
            intrs=self.intrs[:, idx],
            bg_colors=self.bg_colors[:, idx],
            flame_params=flame_params,
            frame_ids=[self.frame_ids[int(i)] for i in indices.cpu().tolist()],
            landmarks_2d=landmarks,
            view_indices=view_indices,
        )

    def to(self, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "MultiViewBatch":
        flame_params = {k: v.to(device=device, dtype=dtype) for k, v in self.flame_params.items()}
        landmarks = self.landmarks_2d
        if landmarks is not None:
            landmarks = landmarks.to(device=device, dtype=dtype)
        view_indices = self.view_indices
        if view_indices is not None:
            view_indices = view_indices.to(device=device)
        return MultiViewBatch(
            images=self.images.to(device=device, dtype=dtype),
            masks=self.masks.to(device=device, dtype=dtype),
            c2ws=self.c2ws.to(device=device, dtype=dtype),
            intrs=self.intrs.to(device=device, dtype=dtype),
            bg_colors=self.bg_colors.to(device=device, dtype=dtype),
            flame_params=flame_params,
            frame_ids=list(self.frame_ids),
            landmarks_2d=landmarks,
            view_indices=view_indices,
        )
