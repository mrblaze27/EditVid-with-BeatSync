#!/usr/bin/env python3
"""
Vision Engine for BeatSync Engine
Provides unified semantic and visual dynamic tagging across all AI tiers:
- Tier 1: 'fast_siglip' - Ultra-fast visual feature & aesthetic dynamics (< 2s, 0 VRAM)
- Tier 2: 'standard_2b' - Qwen3-VL 2B GGUF via llama.cpp Vulkan / CPU
- Tier 3: 'pro_7b'      - Qwen2.5-VL 7B GGUF via llama.cpp Vulkan / CPU
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import cv2
import numpy as np

# Project imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model_manager import ensure_model_available, get_model_paths, AI_VISION_MODELS


# ============================================================================
# FAST VISION ENGINE (Tier 1: < 2s, 0 VRAM)
# ============================================================================

def run_fast_vision_analysis(
    video_path: str,
    candidate_timestamps: List[float],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts deep visual dynamics, color aesthetic variance, motion optical flow,
    edge sharpness, and lighting contrast across candidate frames without LLM overhead.
    """
    if not os.path.exists(video_path) or not candidate_timestamps:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    results: List[Dict[str, Any]] = []
    total = len(candidate_timestamps)

    for idx, t in enumerate(candidate_timestamps):
        if progress_callback and (idx % max(1, total // 10) == 0 or idx == total - 1):
            progress_callback(f"⚡ Fast Vision Analysis: {idx+1}/{total} candidate frames...")

        target_frame_num = int(t * fps)
        if target_frame_num >= total_frames:
            target_frame_num = max(0, total_frames - 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Resize for rapid computation
        h, w = frame.shape[:2]
        small_w = 320
        small_h = int(h * (small_w / float(w)))
        small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)

        # 1. Color Aesthetics & Vibrancy
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mean_sat = float(np.mean(sat)) / 255.0
        mean_val = float(np.mean(val)) / 255.0
        val_std = float(np.std(val)) / 128.0

        # Beauty / Cinematic score (contrast + rich saturation balance)
        beauty_score = min(1.0, max(0.1, (mean_sat * 0.45 + val_std * 0.35 + (1.0 - abs(mean_val - 0.5)) * 0.2)))

        # 2. Visual Complexity & Edge Sharpness
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(1.0, max(0.05, math.log1p(laplacian_var) / 8.0))

        # 3. Action Dynamics (Motion vs neighbor frame if available)
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames - 1, target_frame_num + 3))
        ret_next, frame_next = cap.read()
        if ret_next and frame_next is not None:
            small_next = cv2.resize(frame_next, (small_w, small_h), interpolation=cv2.INTER_AREA)
            gray_next = cv2.cvtColor(small_next, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, gray_next)
            motion_val = float(np.mean(diff)) / 40.0
            action_intensity = min(1.0, max(0.05, motion_val * 0.7 + sharpness_score * 0.3))
        else:
            action_intensity = sharpness_score

        # 4. Semantic Heuristics
        combat_score = min(1.0, max(0.0, action_intensity * 0.85 + (1.0 - beauty_score) * 0.15))
        camera_motion = min(1.0, max(0.0, action_intensity * 0.8))
        character_focus = min(1.0, max(0.1, 0.7 if (mean_sat > 0.35 and sharpness_score > 0.4) else 0.3))
        visual_quality = min(1.0, max(0.3, sharpness_score * 0.6 + beauty_score * 0.4))

        if action_intensity > 0.65:
            emotion = "hype"
            recommended_use = "drop"
        elif beauty_score > 0.65:
            emotion = "soft"
            recommended_use = "soft"
        elif action_intensity > 0.4:
            emotion = "tension"
            recommended_use = "build"
        else:
            emotion = "neutral"
            recommended_use = "flow"

        results.append({
            "timestamp": round(float(t), 3),
            "action_intensity": round(action_intensity, 3),
            "beauty_score": round(beauty_score, 3),
            "combat": round(combat_score, 3),
            "chase": round(action_intensity * 0.7, 3),
            "explosion": round(action_intensity * 0.6 if mean_val > 0.7 else 0.1, 3),
            "character_focus": round(character_focus, 3),
            "camera_motion": round(camera_motion, 3),
            "visual_quality": round(visual_quality, 3),
            "emotion": emotion,
            "recommended_use": recommended_use,
            "description": f"Fast Vision tagged frame at {t:.2f}s (Action: {action_intensity:.2f}, Beauty: {beauty_score:.2f})",
        })

        if progress_callback and (len(results) % 5 == 0 or len(results) == len(candidate_timestamps)):
            sub_pct = int(round(len(results) / max(1, len(candidate_timestamps)) * 100))
            progress_callback(f"Stage 4: Fast Vision analyzing moment {len(results)}/{len(candidate_timestamps)} ({sub_pct}%)...")

    cap.release()
    return results



# ============================================================================
# UNIFIED VISION DISPATCHER
# ============================================================================

def analyze_video_moments_with_tier(
    video_path: str,
    candidate_timestamps: List[float],
    model_tier: str = "standard_2b",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Unified entry point for AI Vision analysis across all 3 tiers.
    """
    if not candidate_timestamps:
        return []

    # Tier 1: Fast Mode (< 2s, 0 VRAM)
    if model_tier == "fast_siglip":
        if progress_callback:
            progress_callback("Running ⚡ Fast Vision Engine (Instant Analysis)...")
        return run_fast_vision_analysis(video_path, candidate_timestamps, progress_callback)

    # Ensure requested model (2B or 7B) is downloaded and ready
    ready = ensure_model_available(model_tier, progress_callback)
    if not ready and model_tier != "fast_siglip":
        if progress_callback:
            progress_callback(f"Model {model_tier} unavailable. Using Fast Vision Engine fallback...")
        return run_fast_vision_analysis(video_path, candidate_timestamps, progress_callback)

    # Tier 2 & 3: Standard (2B) or Cinematic Pro (7B) via Qwen Worker
    model_paths = get_model_paths(model_tier)
    if model_paths.get("model") and os.path.exists(model_paths["model"]):
        os.environ["BEATSYNC_QWEN_LLAMA_MODEL"] = model_paths["model"]
    if model_paths.get("mmproj") and os.path.exists(model_paths["mmproj"]):
        os.environ["BEATSYNC_QWEN_LLAMA_MMPROJ"] = model_paths["mmproj"]

    try:
        from video_analysis import analyze_video_sources
        analysis = analyze_video_sources(
            video_files=[video_path],
            enable_ai=True,
            qwen_model_path=model_paths.get("model"),
        )
        candidates = analysis.get("candidates", [])
        if candidates:
            return candidates
    except Exception as e:
        print(f"   ⚠️  Qwen worker error ({model_tier}), falling back to Fast Vision Engine: {e}")
        if progress_callback:
            progress_callback("AI Model execution error. Falling back to Fast Vision Engine...")

    return run_fast_vision_analysis(video_path, candidate_timestamps, progress_callback)

