#!/usr/bin/env python3
"""
Download and convert CoreML models for the VoxelKit neural pipeline.

Models:
1. Depth Anything V2 Small — fast relative depth (~30ms ANE)
2. Apple Depth Pro — accurate metric depth (~200ms)
3. YOLOv8n — lightweight object detection

Usage:
    pip install coremltools torch torchvision huggingface_hub ultralytics
    python3 Scripts/download_models.py [--output-dir Models]

Output:
    Models/DepthAnythingV2Small.mlmodelc
    Models/DepthPro.mlmodelc
    Models/YOLOv8n.mlmodelc
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    missing = []
    for pkg in ["coremltools", "torch", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install coremltools torch torchvision huggingface_hub ultralytics")
        sys.exit(1)


def download_depth_anything_v2(output_dir: Path):
    """Download Depth Anything V2 Small CoreML model from HuggingFace."""
    import coremltools as ct
    from huggingface_hub import hf_hub_download

    model_name = "DepthAnythingV2Small"
    mlpackage_path = output_dir / f"{model_name}.mlpackage"
    mlmodelc_path = output_dir / f"{model_name}.mlmodelc"

    if mlmodelc_path.exists():
        print(f"  [skip] {model_name}.mlmodelc already exists")
        return

    print(f"  Downloading {model_name} from HuggingFace...")

    # Try pre-converted CoreML model first
    try:
        downloaded = hf_hub_download(
            repo_id="apple/coreml-depth-anything-v2-small",
            filename="DepthAnythingV2Small.mlpackage/Metadata.json",
            local_dir=str(output_dir / "_hf_cache"),
        )
        hf_cache_dir = output_dir / "_hf_cache"

        # The HF repo has the full .mlpackage directory
        hf_mlpackage = hf_cache_dir / "DepthAnythingV2Small.mlpackage"
        if hf_mlpackage.exists():
            print(f"  Found pre-converted .mlpackage, compiling...")
            model = ct.models.MLModel(str(hf_mlpackage))
            model.save(str(mlpackage_path))
            _compile_mlpackage(mlpackage_path, mlmodelc_path)
            return
    except Exception as e:
        print(f"  Pre-converted model not available ({e}), converting from PyTorch...")

    # Fallback: convert from PyTorch
    try:
        import torch
        from huggingface_hub import hf_hub_download

        # Download the PyTorch checkpoint
        ckpt_path = hf_hub_download(
            repo_id="depth-anything/Depth-Anything-V2-Small",
            filename="depth_anything_v2_vits.pth",
            local_dir=str(output_dir / "_hf_cache"),
        )

        print(f"  Converting to CoreML (input: 518x518)...")
        # Use coremltools trace-based conversion
        # The model expects 1x3x518x518 input
        _convert_depth_anything_v2_from_torch(ckpt_path, mlpackage_path)
        _compile_mlpackage(mlpackage_path, mlmodelc_path)

    except Exception as e:
        print(f"  [WARN] Could not convert Depth Anything V2: {e}")
        print(f"  Please manually place {model_name}.mlmodelc in {output_dir}")


def _convert_depth_anything_v2_from_torch(ckpt_path: str, output_path: Path):
    """Convert Depth Anything V2 Small from PyTorch to CoreML."""
    import torch
    import coremltools as ct

    # Minimal wrapper that loads the model
    # Depth Anything V2 uses a DINOv2-based encoder
    try:
        sys.path.insert(0, str(Path(ckpt_path).parent))
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    except ImportError:
        # If the model code isn't available, use huggingface transformers
        from transformers import AutoModelForDepthEstimation
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small"
        )

    model.eval()
    example_input = torch.randn(1, 3, 518, 518)

    traced = torch.jit.trace(model, example_input)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 518, 518),
                             scale=1.0/255.0, bias=[0, 0, 0])],
        outputs=[ct.TensorType(name="depth")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.save(str(output_path))


def download_depth_pro(output_dir: Path):
    """Download Apple Depth Pro CoreML model."""
    model_name = "DepthPro"
    mlmodelc_path = output_dir / f"{model_name}.mlmodelc"

    if mlmodelc_path.exists():
        print(f"  [skip] {model_name}.mlmodelc already exists")
        return

    print(f"  Downloading {model_name}...")
    print(f"  NOTE: Apple Depth Pro requires manual download from:")
    print(f"    https://github.com/apple/ml-depth-pro")
    print(f"  Steps:")
    print(f"    1. git clone https://github.com/apple/ml-depth-pro")
    print(f"    2. cd ml-depth-pro && pip install -e .")
    print(f"    3. python3 -c \"")
    print(f"       import depth_pro")
    print(f"       model, transform = depth_pro.create_model_and_transforms()")
    print(f"       # Export to CoreML using coremltools")
    print(f"       \"")
    print(f"    4. Place {model_name}.mlmodelc in {output_dir}")
    print()

    # Try automated conversion if ml-depth-pro is installed
    try:
        import depth_pro
        import torch
        import coremltools as ct

        print(f"  Found depth_pro, attempting conversion...")
        model, transform = depth_pro.create_model_and_transforms()
        model.eval()

        example_input = torch.randn(1, 3, 1536, 1536)
        traced = torch.jit.trace(model, example_input)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.ImageType(name="image", shape=(1, 3, 1536, 1536),
                                 scale=1.0/255.0, bias=[0, 0, 0])],
            outputs=[ct.TensorType(name="depth")],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
        )

        mlpackage_path = output_dir / f"{model_name}.mlpackage"
        mlmodel.save(str(mlpackage_path))
        _compile_mlpackage(mlpackage_path, mlmodelc_path)

    except ImportError:
        print(f"  [WARN] depth_pro not installed. Manual setup required.")
    except Exception as e:
        print(f"  [WARN] Conversion failed: {e}")


def download_yolov8n(output_dir: Path):
    """Download and convert YOLOv8n to CoreML."""
    model_name = "YOLOv8n"
    mlmodelc_path = output_dir / f"{model_name}.mlmodelc"

    if mlmodelc_path.exists():
        print(f"  [skip] {model_name}.mlmodelc already exists")
        return

    print(f"  Downloading {model_name}...")

    try:
        from ultralytics import YOLO

        # Download and export to CoreML
        model = YOLO("yolov8n.pt")
        export_path = model.export(format="coreml", nms=True, imgsz=640)

        # ultralytics exports as .mlpackage
        mlpackage_path = Path(export_path)
        if mlpackage_path.exists():
            _compile_mlpackage(mlpackage_path, mlmodelc_path)
        else:
            print(f"  [WARN] Export produced unexpected path: {export_path}")

    except ImportError:
        print(f"  [WARN] ultralytics not installed.")
        print(f"  Install with: pip install ultralytics")
        print(f"  Then run: yolo export model=yolov8n.pt format=coreml")
    except Exception as e:
        print(f"  [WARN] YOLOv8n conversion failed: {e}")


def _compile_mlpackage(mlpackage_path: Path, mlmodelc_path: Path):
    """Compile .mlpackage to .mlmodelc using xcrun coremlcompiler."""
    print(f"  Compiling {mlpackage_path.name} → {mlmodelc_path.name}...")
    result = subprocess.run(
        ["xcrun", "coremlcompiler", "compile",
         str(mlpackage_path), str(mlmodelc_path.parent)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [WARN] coremlcompiler failed: {result.stderr}")
        print(f"  Fallback: keeping .mlpackage (CoreML can load it at runtime)")
    else:
        print(f"  Compiled: {mlmodelc_path}")


def main():
    parser = argparse.ArgumentParser(description="Download CoreML models for VoxelKit")
    parser.add_argument("--output-dir", default="Models",
                        help="Output directory for .mlmodelc files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    check_dependencies()

    print("[1/3] Depth Anything V2 Small")
    download_depth_anything_v2(output_dir)
    print()

    print("[2/3] Apple Depth Pro")
    download_depth_pro(output_dir)
    print()

    print("[3/3] YOLOv8 Nano")
    download_yolov8n(output_dir)
    print()

    # Summary
    print("=" * 50)
    print("Summary:")
    for name in ["DepthAnythingV2Small", "DepthPro", "YOLOv8n"]:
        mlmodelc = output_dir / f"{name}.mlmodelc"
        mlpackage = output_dir / f"{name}.mlpackage"
        if mlmodelc.exists():
            print(f"  ✓ {name}.mlmodelc")
        elif mlpackage.exists():
            print(f"  ~ {name}.mlpackage (not compiled)")
        else:
            print(f"  ✗ {name} — manual setup needed")

    print(f"\nPlace models in your app's Resources or Application Support/VoxelUI/Models/")
    print(f"NeuralPipeline.loadModels(from:) will load them at runtime.")


if __name__ == "__main__":
    main()
