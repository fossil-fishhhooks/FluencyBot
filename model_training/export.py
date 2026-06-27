import argparse
import os
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, AutoTokenizer


parser = argparse.ArgumentParser(description="Export a trained fluency model to ONNX with INT8 quantization.")
parser.add_argument("--model-dir",   type=str, required=True,  help="Path to saved model folder (e.g. ./fluency_model)")
parser.add_argument("--output",      type=str, default=None,   help="Output .onnx path (default: <model-dir>/fluency.onnx)")
parser.add_argument("--max-tokens",  type=int, default=64,     help="Max token sequence length (default: 64)")
parser.add_argument("--no-quantize", action="store_true",      help="Skip INT8 quantization")
args = parser.parse_args()

output_path = args.output or os.path.join(args.model_dir, "fluency.onnx")
quant_path  = output_path.replace(".onnx", ".quant.onnx")


print(f"Loading model from {args.model_dir} ...")
model     = BertForSequenceClassification.from_pretrained(args.model_dir)
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model.eval()
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# check
dummy = tokenizer(
    "The cat sat on the mat.",
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=args.max_tokens,
)
input_ids      = dummy["input_ids"]
attention_mask = dummy["attention_mask"]


class FluentWrapper(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, input_ids, attention_mask):
        return self.base(input_ids=input_ids, attention_mask=attention_mask).logits

wrapper = FluentWrapper(model)
wrapper.eval()

print(f"\nExporting FP32 ONNX to {output_path} ...")
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits":         {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,  # force legacy TorchScript exporter. dynamo path breaks BERT
    )

fp32_size = os.path.getsize(output_path) / 1024 / 1024
print(f"  FP32 size: {fp32_size:.1f} MB")



if not args.no_quantize:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        print(f"\nQuantizing to {quant_path} ...")
        quantize_dynamic(
            model_input=output_path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
        )
        q_size = os.path.getsize(quant_path) / 1024 / 1024
        print(f"  INT8 size: {q_size:.1f} MB  ({fp32_size/q_size:.1f}x smaller)")
    except ImportError:
        print("\nonnxruntime not installed: skipping quantization.")
        print("Run: pip install onnxruntime")
        quant_path = None
else:
    quant_path = None

#verify
try:
    import onnxruntime as ort
    import numpy as np

    verify_path = quant_path if (quant_path and os.path.exists(quant_path)) else output_path
    print(f"\nVerifying {os.path.basename(verify_path)} with onnxruntime ...")
    sess  = ort.InferenceSession(verify_path, providers=["CPUExecutionProvider"])
    out   = sess.run(["logits"], {
        "input_ids":      input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    })
    score = float(1 / (1 + np.exp(-out[0][0][0])))
    print(f"  'The cat sat on the mat.' --> {score:.4f}  (expected ~0.98)")
    print("  Verification OK" if score > 0.5 else "  WARNING: score unexpectedly low")
except ImportError:
    print("\nonnxruntime not installed: skipping verification.")


print(f"\n{'='*50}")
print(f"Model dir  : {args.model_dir}")
print(f"FP32 ONNX  : {output_path}  ({fp32_size:.1f} MB)")
if quant_path and os.path.exists(quant_path):
    print(f"INT8 ONNX  : {quant_path}  ({os.path.getsize(quant_path)/1024/1024:.1f} MB)")