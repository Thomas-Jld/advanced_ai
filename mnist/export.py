from mnist.model import superdupermodel
from onnx_tf.backend import prepare
from tensorflowjs.converters.tf_saved_model_conversion_v2 import (
    convert_tf_saved_model,
)

import onnx
import torch

def export(path: str) -> None:
    torch_path = f"{path}.pth"
    onnx_path = f"{path}.onnx"
    tensorflow_path = f"{path}.pb"
    tensorflowjs_path = path

    model = superdupermodel()
    model.load_state_dict(torch.load(torch_path))
    model = model.eval()
    
    torch.onnx.export(
        model,
        torch.zeros((1, 1, 28, 28)),
        onnx_path,
        do_constant_folding=True,
        export_params=True,
        input_names=["img"],
        output_names=["preds"],
        opset_version=10,
        verbose=True,
    )

    model = onnx.load(onnx_path)
    prepare(model).export_graph(tensorflow_path)
    convert_tf_saved_model(tensorflow_path, tensorflowjs_path)