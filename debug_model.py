from ultralytics import YOLO
import torch

# Load model
model = YOLO("models/fishsegnet_prl.yaml").model

# Input giả
x = torch.randn(1, 3, 640, 640)

print("=" * 80)
print("DEBUG MODEL")
print("=" * 80)

# Hook function
def hook_fn(name):
    def hook(module, input, output):

        # input shape
        if isinstance(input, tuple):
            in_shape = [i.shape for i in input if hasattr(i, "shape")]
        else:
            in_shape = input.shape

        # output shape
        if hasattr(output, "shape"):
            out_shape = output.shape
        elif isinstance(output, (list, tuple)):
            out_shape = [o.shape for o in output if hasattr(o, "shape")]
        else:
            out_shape = "unknown"

        print(f"\n{name}")
        print(f"INPUT : {in_shape}")
        print(f"OUTPUT: {out_shape}")

    return hook

# Gắn hook vào từng layer
for i, layer in enumerate(model.model):
    layer.register_forward_hook(hook_fn(f"Layer {i}: {layer.__class__.__name__}"))

# Chạy forward
with torch.no_grad():
    y = model(x)

print("\nDONE")