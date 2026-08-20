#!/usr/bin/env python3
"""
Model Manager for BeatSync Engine
Handles local AI model discovery, verification, and on-demand streaming downloads.
Supports:
- Fast Mode: SigLIP / OpenCV Visual Feature Engine (0 VRAM, ~1s)
- Standard Mode: Qwen3-VL 2B GGUF (~2.5 GB VRAM)
- Cinematic Pro Mode: Qwen2.5-VL 7B GGUF (~6 GB VRAM)
"""

import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT_DIR / "bin"
MODELS_DIR = BIN_DIR / "models"


# Model Registry Definition
AI_VISION_MODELS = {
    "fast_siglip": {
        "name": "⚡ Fast Mode (Zero-Shot / Visual Dynamics)",
        "tier": "fast_siglip",
        "vram_mb": 0,
        "description": "Ultra-fast (<2s) visual motion, color aesthetic, and beat-impact scoring. Runs on any CPU/GPU with zero VRAM.",
        "requires_download": False,
        "files": {},
    },
    "standard_2b": {
        "name": "🚀 Standard Mode (Qwen3-VL 2B - Balanced)",
        "tier": "standard_2b",
        "vram_mb": 2500,
        "description": "Balanced high-quality semantic vision model. Identifies action, combat, aesthetics, and emotion.",
        "requires_download": True,
        "files": {
            "model": {
                "filename": "Qwen3VL-2B-Instruct-Q8_0.gguf",
                "url": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/qwen2.5-vl-3b-instruct-q8_0.gguf",
                "size_mb": 1834,
            },
            "mmproj": {
                "filename": "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
                "url": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/mmproj-qwen2.5-vl-3b-instruct-f16.gguf",
                "size_mb": 819,
            },
        },
    },
    "pro_7b": {
        "name": "👑 Cinematic Pro (Qwen2.5-VL 7B - Deep Semantic HQ)",
        "tier": "pro_7b",
        "vram_mb": 6000,
        "description": "State-of-the-art vision-language intelligence for high-end GPUs. Deep scene composition and cinematic combat tagging.",
        "requires_download": True,
        "files": {
            "model": {
                "filename": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "url": "https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "size_mb": 4680,
            },
            "mmproj": {
                "filename": "mmproj-Qwen2.5-VL-7B-Instruct-F16.gguf",
                "url": "https://huggingface.co/bartowski/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/mmproj-Qwen2.5-VL-7B-Instruct-F16.gguf",
                "size_mb": 840,
            },
        },
    },
}


def get_models_dir() -> Path:
    """Return and create the models directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


def check_model_status(tier: str = "standard_2b") -> Tuple[bool, str]:
    """Check if all necessary files for the specified model tier exist locally."""
    if tier not in AI_VISION_MODELS:
        return False, f"Unknown model tier: {tier}"

    info = AI_VISION_MODELS[tier]
    if not info.get("requires_download"):
        return True, "Ready (built-in)"

    models_dir = get_models_dir()
    for file_key, file_info in info.get("files", {}).items():
        fname = file_info["filename"]
        target_path = models_dir / fname
        if not target_path.exists() or target_path.stat().st_size < 1024 * 1024:
            return False, f"Missing {fname} ({file_info.get('size_mb', 0)} MB)"

    return True, "Ready (Installed)"


def get_model_paths(tier: str = "standard_2b") -> Dict[str, str]:
    """Return absolute file paths for the requested model tier."""
    models_dir = get_models_dir()
    if tier == "pro_7b":
        return {
            "model": str(models_dir / AI_VISION_MODELS["pro_7b"]["files"]["model"]["filename"]),
            "mmproj": str(models_dir / AI_VISION_MODELS["pro_7b"]["files"]["mmproj"]["filename"]),
        }
    elif tier == "standard_2b":
        return {
            "model": str(models_dir / AI_VISION_MODELS["standard_2b"]["files"]["model"]["filename"]),
            "mmproj": str(models_dir / AI_VISION_MODELS["standard_2b"]["files"]["mmproj"]["filename"]),
        }
    return {}


def download_file_with_progress(
    url: str,
    output_path: Path,
    expected_size_mb: int = 0,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """Download a file with streaming chunk progress reporting."""
    temp_path = output_path.with_suffix(".tmp_download")
    try:
        if progress_callback:
            progress_callback(f"Connecting to download server for {output_path.name}...")

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "BeatSync-Engine/1.2 (Windows; x64)"}
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.info().get("Content-Length", 0))
            if total_size <= 0 and expected_size_mb > 0:
                total_size = expected_size_mb * 1024 * 1024

            downloaded = 0
            block_size = 1024 * 1024  # 1 MB chunks
            start_time = time.time()
            last_notify = 0

            with open(temp_path, "wb") as f_out:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_notify > 0.5:  # Throttle notifications to twice per second
                        last_notify = now
                        elapsed = max(0.1, now - start_time)
                        speed_mbps = (downloaded / (1024 * 1024)) / elapsed
                        if total_size > 0:
                            percent = min(100.0, (downloaded / total_size) * 100.0)
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            msg = f"Downloading {output_path.name}: {percent:.1f}% ({mb_done:.1f}/{mb_total:.1f} MB) @ {speed_mbps:.1f} MB/s"
                        else:
                            mb_done = downloaded / (1024 * 1024)
                            msg = f"Downloading {output_path.name}: {mb_done:.1f} MB @ {speed_mbps:.1f} MB/s"

                        if progress_callback:
                            progress_callback(msg)

        # Atomic rename on complete
        if temp_path.exists():
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
            return True
        return False

    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        if progress_callback:
            progress_callback(f"Download failed for {output_path.name}: {e}")
        print(f"   ⚠️ Download error for {output_path.name}: {e}")
        return False


def ensure_model_available(
    tier: str = "standard_2b",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> bool:
    """
    Ensure the requested model tier is ready.
    If files are missing, automatically downloads them on-demand.
    """
    if tier == "fast_siglip":
        return True

    is_ready, status_msg = check_model_status(tier)
    if is_ready:
        return True

    info = AI_VISION_MODELS.get(tier)
    if not info:
        return False

    models_dir = get_models_dir()
    if progress_callback:
        progress_callback(f"Preparing AI Vision Model ({info['name']})...")

    for file_key, file_info in info.get("files", {}).items():
        fname = file_info["filename"]
        target_path = models_dir / fname
        if not target_path.exists() or target_path.stat().st_size < 1024 * 1024:
            if progress_callback:
                progress_callback(f"Starting on-demand download for {fname} (~{file_info['size_mb']} MB)...")
            success = download_file_with_progress(
                url=file_info["url"],
                output_path=target_path,
                expected_size_mb=file_info.get("size_mb", 0),
                progress_callback=progress_callback,
            )
            if not success:
                # Fallback to standard_2b if pro_7b download fails
                if tier == "pro_7b":
                    if progress_callback:
                        progress_callback("Cinematic Pro download failed. Falling back to Standard Mode (2B)...")
                    return ensure_model_available("standard_2b", progress_callback)
                return False

    return True
