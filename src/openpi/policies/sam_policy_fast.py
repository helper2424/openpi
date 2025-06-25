import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_sam_example() -> dict:
    # TODO переделать! Тк не уверен что такие нужно ключи
    """Creates a random input example for the SAM policy."""
    return {
        "laptop": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "phone": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "side": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "state": np.ones((7,)),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SAMInputs(transforms.DataTransformFn):

    action_dim: int
    model_type: _model.ModelType = _model.ModelType.PI0_FAST

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0_FAST
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["laptop"])
        wrist_image = _parse_image(data["phone"])
        side_image = _parse_image(data["side"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Since there is no right wrist, replace with zeros
                "right_wrist_0_rgb": side_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Since the "slot" for the right wrist is not used, this mask is set
                # to False
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            # inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = transforms.pad_to_dim(np.asarray(data["actions"]), self.action_dim)
        
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SAMOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        return {"actions": np.asarray(data["actions"][:, :7])}
