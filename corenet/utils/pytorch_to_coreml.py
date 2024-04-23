#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict, Optional, Tuple, Union

import coremltools as ct
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.mobile_optimizer import optimize_for_mobile
from torchvision.transforms import functional as F

from corenet.utils import logger
from corenet.utils.tensor_utils import create_rand_tensor


def convert_pytorch_to_coreml(
    opts,
    pytorch_model: torch.nn.Module,
    jit_model_only: Optional[bool] = False,
    convert_to: str = "neuralnetwork",
    *args,
    **kwargs
) -> Dict:
    """
    Convert Pytorch model to CoreML

    Args:
        pytorch_model: Pytorch model that needs to be converted to JIT or CoreML
        jit_model_only: If set, do not create the optimized or CoreML model.
        convert_to: If 'neuralnetwork', convert to espresso format. If 'mlpackage',
            convert to the MIL format.

    Returns:
        A dict containing the JIT model, the optimized model, and the CoreML model.
    """

    input_image_path = getattr(opts, "conversion.input_image_path", None)
    if input_image_path is not None:
        input_pil_img = Image.open(input_image_path).convert("RGB")
        input_pil_img = F.resize(
            img=input_pil_img, size=256, interpolation=F.InterpolationMode.BILINEAR
        )
        input_pil_img = F.center_crop(img=input_pil_img, output_size=224)
        input_tensor = F.pil_to_tensor(input_pil_img).float()
        input_tensor.div_(255.0)
        input_tensor = input_tensor.unsqueeze(0)  # add dummy batch dimension
        input_tuple = (input_tensor,)
    elif hasattr(pytorch_model, "conversion_inputs"):
        input_pil_img = None
        input_tuple = pytorch_model.conversion_inputs()
    else:
        input_pil_img = None
        input_tensor = create_rand_tensor(opts=opts, device="cpu")
        input_tuple = (input_tensor,)

    if pytorch_model.training:
        pytorch_model.eval()

    # Prepare model to be exported (only if implemented)
    if hasattr(pytorch_model, "get_exportable_model"):
        logger.log("Preparing model for export.")
        pytorch_model = pytorch_model.get_exportable_model()

    with torch.no_grad():
        pytorch_out = pytorch_model(*input_tuple)

        jit_model = torch.jit.trace(pytorch_model, input_tuple)
        jit_out = jit_model(*input_tuple)
        assertion_check(py_out=pytorch_out, jit_out=jit_out)

        jit_model_optimized = optimize_for_mobile(jit_model)
        jit_optimized_out = jit_model_optimized(*input_tuple)
        assertion_check(py_out=pytorch_out, jit_out=jit_optimized_out)

        if jit_model_only and torch.cuda.device_count() > 0:
            # For inference on GPU
            return {"coreml": None, "jit": jit_model, "jit_optimized": None}
        elif jit_model_only and torch.cuda.device_count() == 0:
            # For inference on CPU
            return {"coreml": None, "jit": jit_model_optimized, "jit_optimized": None}

        if hasattr(pytorch_model, "ct_convert_inputs_outputs_types"):
            inputs, outputs = pytorch_model.ct_convert_inputs_outputs_types(input_tuple)
        else:
            inputs = [
                ct.ImageType(
                    name="input", shape=input_tuple[0].shape, scale=1.0 / 255.0
                )
            ]
            outputs = None

        coreml_model = ct.convert(
            model=jit_model,
            inputs=inputs,
            convert_to=convert_to,
            outputs=outputs,
        )

        if input_pil_img is not None:
            out = coreml_model.predict({"input": input_pil_img})

        return {
            "coreml": coreml_model,
            "jit": jit_model,
            "jit_optimized": jit_model_optimized,
        }


def assertion_check(
    py_out: Union[Tensor, Dict, Tuple], jit_out: Union[Tensor, Dict, Tuple]
) -> None:
    if isinstance(py_out, Dict):
        assert isinstance(jit_out, Dict)
        keys = py_out.keys()
        for k in keys:
            np.testing.assert_almost_equal(
                py_out[k].cpu().numpy(),
                jit_out[k].cpu().numpy(),
                decimal=3,
                verbose=True,
            )
    elif isinstance(py_out, Tensor):
        assert isinstance(jit_out, Tensor)
        np.testing.assert_almost_equal(
            py_out.cpu().numpy(), jit_out.cpu().numpy(), decimal=3, verbose=True
        )
    elif isinstance(py_out, Tuple):
        assert isinstance(jit_out, Tuple)
        for x, y in zip(py_out, jit_out):
            np.testing.assert_almost_equal(
                x.cpu().numpy(), y.cpu().numpy(), decimal=3, verbose=True
            )

    else:
        raise NotImplementedError(
            "Only Dictionary[Tensors] or Tuple[Tensors] or Tensors are supported as outputs"
        )
