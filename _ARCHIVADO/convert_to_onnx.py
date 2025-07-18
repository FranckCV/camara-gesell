import torch
import os
import sys

# Añadir ruta al módulo
sys.path.append(os.path.join(os.path.dirname(__file__), 'emonet'))
from models.emonet import EmoNet

n_expression = 8
model_path = 'emonet_8.pth'
output_path = 'emonet_8.onnx'

# Cargar el modelo
model = EmoNet(n_expression=n_expression)
state_dict = torch.load(model_path, map_location='cpu')
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=False)
model.eval()

# Entrada simulada
dummy_input = torch.randn(1, 3, 256, 256)

# Exportar a ONNX
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['expression']
)

print(f"✅ Modelo exportado correctamente como {output_path}")
