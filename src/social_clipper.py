#!/usr/bin/env python3
"""
Social Clipper Module - Vertical 9:16 AI Highlights Generator
Automates the extraction of viral TikTok / Reels / Shorts clips from full videos.
- Intelligent AI (Qwen3-VL) & Audio-Visual highlight scoring
- Beat-locked and scene-aligned boundaries
- High-quality 9:16 vertical framing (Smart Crop, Blurred Ambient Fill, Letterbox)
- Hardware GPU NVENC acceleration with CPU fallback
"""

import os
import sys
import time
import math
import tempfile
import shutil
import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2

# Project imports
from logger import setup_environment, ROOT_DIR
setup_environment()

from gpu_cpu_utils import GPU_AVAILABLE, NVENC_AVAILABLE, PARALLEL_WORKERS, MAX_THREADS
from paths import get_processing_dir, get_shorts_output_dir
from ffmpeg_processing import (
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    detect_video_scene_changes,
    detect_video_keyframes,
    has_audio_stream,
    extract_audio_from_video,
    extract_vertical_clip_9_16,
)

# Constants
DEFAULT_TARGET_WIDTH = 1080
DEFAULT_TARGET_HEIGHT = 1920


def _fmt_seconds(seconds: float) -> str:
    try:
        val = float(seconds)
    except Exception:
        val = 0.0
    m, s = divmod(val, 60)
    if m > 0:
        return f"{int(m):02d}m{s:04.1f}s"
    return f"{s:04.1f}s"


def _fmt_time_badge(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"


def _clamp(val: Any, lo: float = 0.0, hi: float = 1.0, default: float = 0.0) -> float:
    try:
        v = float(val)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


class SocialClipResult:
    def __init__(self,
                 clip_index: int,
                 file_path: str,
                 start_time: float,
                 end_time: float,
                 duration: float,
                 viral_score: float,
                 label: str,
                 framing_mode: str,
                 ai_tags: Dict[str, Any] = None):
        self.clip_index = clip_index
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.viral_score = viral_score
        self.label = label
        self.framing_mode = framing_mode
        self.ai_tags = ai_tags or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_index": self.clip_index,
            "file_path": self.file_path,
            "filename": self.filename,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "viral_score": self.viral_score,
            "label": self.label,
            "framing_mode": self.framing_mode,
            "ai_tags": self.ai_tags,
        }


def _analyze_audio_timeline(audio_path: str, duration: float) -> Dict[str, Any]:
    """Analyze audio rhythm, RMS energy wave, tempo, and beat grid using librosa."""
    try:
        import librosa
        sr = 22050
        hop_length = 512
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
        if y.size == 0:
            return {}

        y_norm = librosa.util.normalize(y)
        tempo_val, beat_frames = librosa.beat.beat_track(y=y_norm, sr=sr, hop_length=hop_length)
        if isinstance(tempo_val, np.ndarray):
            tempo = float(tempo_val.item()) if tempo_val.size > 0 else 120.0
        else:
            tempo = float(tempo_val)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        # Compute RMS energy curve
        rms_raw = librosa.feature.rms(y=y_norm, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms_raw)), sr=sr, hop_length=hop_length)

        # Normalize RMS (0.0 to 1.0)
        p2, p98 = np.percentile(rms_raw, 2), np.percentile(rms_raw, 98)
        if p98 - p2 > 1e-6:
            rms_norm = np.clip((rms_raw - p2) / (p98 - p2), 0.0, 1.0)
        else:
            rms_norm = np.full_like(rms_raw, 0.5)

        # Smooth energy wave (approx 2.0s rolling window)
        smooth_frames = max(3, int((2.0 * sr) / hop_length))
        if smooth_frames % 2 == 0:
            smooth_frames += 1
        pad = smooth_frames // 2
        padded = np.pad(rms_norm, (pad, pad), mode="edge")
        kernel = np.ones(smooth_frames) / smooth_frames
        wave = np.convolve(padded, kernel, mode="valid")[:len(rms_norm)]

        # Onset / Drop impact score
        onset_env = librosa.onset.onset_strength(y=y_norm, sr=sr, hop_length=hop_length)
        onset_p2, onset_p98 = np.percentile(onset_env, 2), np.percentile(onset_env, 98)
        if onset_p98 - onset_p2 > 1e-6:
            onset_norm = np.clip((onset_env - onset_p2) / (onset_p98 - onset_p2), 0.0, 1.0)
        else:
            onset_norm = np.full_like(onset_env, 0.5)

        return {
            "has_audio": True,
            "tempo": tempo,
            "beat_times": beat_times,
            "rms_times": rms_times,
            "rms_norm": rms_norm,
            "wave": wave,
            "onset_norm": onset_norm,
        }
    except Exception as e:
        print(f"   ⚠️  Audio analysis warning: {e}")
        return {"has_audio": False}


def _measure_visual_motion_sampled(video_path: str, sample_interval: float = 0.5) -> List[Dict[str, float]]:
    """Sample video frames to measure motion intensity and visual sharpness."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else 0.0
    if duration <= 0:
        cap.release()
        return []

    step_frames = max(1, int(round(fps * sample_interval)))
    results: List[Dict[str, float]] = []

    prev_gray = None
    prev_time = 0.0

    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cur_time = frame_idx / fps
        h, w = frame.shape[:2]
        # Resize small for fast analysis
        small = cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Measure sharpness / detail using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = _clamp(laplacian_var / 500.0, 0.0, 1.0)

        # Measure motion
        motion_score = 0.5
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = _clamp(np.mean(diff) / 35.0, 0.0, 1.0)

        results.append({
            "time": cur_time,
            "motion": motion_score,
            "sharpness": sharpness_score,
        })

        prev_gray = gray
        prev_time = cur_time
        frame_idx += step_frames

    cap.release()
    return results


def _get_qwen_ai_tags_for_video(video_path: str,
                                scene_changes: List[float],
                                duration: float,
                                enable_ai: bool = True) -> List[Dict[str, Any]]:
    """Fetch or run Qwen3-VL tags for candidate moments in the video."""
    if not enable_ai:
        return []

    try:
        from video_analysis import (
            DEFAULT_QWEN_MODEL_DIR,
            _cache_path,
            _load_cache,
            _qwen_backend_available,
            analyze_video_sources,
        )

        # Check existing cache
        cache_p = _cache_path(video_path, enable_ai=True, qwen_model_path=DEFAULT_QWEN_MODEL_DIR)
        cached_data = _load_cache(cache_p, require_ai=True)
        if cached_data and cached_data.get("candidates"):
            return cached_data["candidates"]

        # If Qwen backend available, run quick analysis
        if _qwen_backend_available(DEFAULT_QWEN_MODEL_DIR):
            analysis = analyze_video_sources(
                video_files=[video_path],
                audio_profile=None,
                use_gpu=GPU_AVAILABLE,
                enable_ai=True,
                qwen_model_path=DEFAULT_QWEN_MODEL_DIR,
            )
            if analysis and analysis.get("candidates"):
                return analysis["candidates"]
    except Exception as e:
        print(f"   ⚠️  Qwen AI video tagging skipped: {e}")

    return []


def _interpolate_time_series(target_time: float, times: np.ndarray, values: np.ndarray, default: float = 0.5) -> float:
    if times is None or len(times) == 0 or values is None or len(values) == 0:
        return default
    if target_time <= times[0]:
        return float(values[0])
    if target_time >= times[-1]:
        return float(values[-1])
    return float(np.interp(target_time, times, values))


def _calculate_social_score_curve(
    duration: float,
    audio_info: Dict[str, Any],
    visual_samples: List[Dict[str, float]],
    ai_candidates: List[Dict[str, Any]],
    scene_changes: List[float],
    ai_strategy: str = "smart_viral",
    resolution_step: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Build a continuous second-by-second Social Engagement/Viral Score curve.
    """
    timeline_steps = np.arange(0.0, max(0.5, duration), resolution_step)
    scores = np.zeros_like(timeline_steps, dtype=float)
    step_details: List[Dict[str, Any]] = []

    has_audio = audio_info.get("has_audio", False)
    rms_times = audio_info.get("rms_times", np.array([]))
    rms_norm = audio_info.get("rms_norm", np.array([]))
    wave = audio_info.get("wave", np.array([]))
    onset_norm = audio_info.get("onset_norm", np.array([]))

    vis_times = np.array([s["time"] for s in visual_samples]) if visual_samples else np.array([])
    vis_motions = np.array([s["motion"] for s in visual_samples]) if visual_samples else np.array([])
    vis_sharpness = np.array([s["sharpness"] for s in visual_samples]) if visual_samples else np.array([])

    # Weightings based on strategy
    if ai_strategy == "peak_energy_drop":
        w_audio_wave = 0.40
        w_audio_onset = 0.35
        w_motion = 0.15
        w_ai = 0.10
    elif ai_strategy == "visual_action":
        w_audio_wave = 0.20
        w_audio_onset = 0.15
        w_motion = 0.35
        w_ai = 0.30
    elif ai_strategy == "cinematic_beauty":
        w_audio_wave = 0.20
        w_audio_onset = 0.10
        w_motion = 0.15
        w_ai = 0.55
    else:
        # Default 'smart_viral': balanced high engagement
        w_audio_wave = 0.25
        w_audio_onset = 0.25
        w_motion = 0.25
        w_ai = 0.25

    scene_set = set(round(s, 2) for s in scene_changes)

    for i, t in enumerate(timeline_steps):
        # 1. Audio Score
        if has_audio and len(rms_times) > 0:
            a_wave = _interpolate_time_series(t, rms_times, wave, 0.5)
            a_onset = _interpolate_time_series(t, rms_times, onset_norm, 0.5)
        else:
            a_wave = 0.5
            a_onset = 0.5

        # 2. Visual Score
        if len(vis_times) > 0:
            v_motion = _interpolate_time_series(t, vis_times, vis_motions, 0.5)
            v_sharp = _interpolate_time_series(t, vis_times, vis_sharpness, 0.5)
        else:
            v_motion = 0.5
            v_sharp = 0.5

        # 3. AI Score
        ai_score = 0.5
        matched_ai = None
        for cand in ai_candidates:
            c_start = float(cand.get("start", 0.0))
            c_end = float(cand.get("end", c_start))
            if c_start <= t <= c_end:
                matched_ai = cand
                # Calculate AI composite
                action = _clamp(cand.get("action_intensity") or cand.get("combat") or 0.5)
                beauty = _clamp(cand.get("beauty_score") or cand.get("visual_quality") or 0.5)
                emotion = str(cand.get("emotion", "neutral")).lower()
                rec_use = str(cand.get("recommended_use", "flow")).lower()

                emotion_bonus = 0.2 if emotion in ("hype", "tension") else (0.1 if emotion == "soft" else 0.0)
                use_bonus = 0.25 if rec_use in ("drop", "build") else 0.0

                if ai_strategy == "cinematic_beauty":
                    ai_score = _clamp(0.6 * beauty + 0.2 * action + emotion_bonus + use_bonus)
                elif ai_strategy == "visual_action":
                    ai_score = _clamp(0.6 * action + 0.2 * beauty + emotion_bonus + use_bonus)
                else:
                    ai_score = _clamp(0.4 * action + 0.3 * beauty + emotion_bonus + use_bonus)
                break

        # Scene cut bonus (transitional dynamism)
        scene_bonus = 0.05 if any(abs(t - sc) <= 0.3 for sc in scene_set) else 0.0

        # Unified combined score
        total_score = (
            w_audio_wave * a_wave +
            w_audio_onset * a_onset +
            w_motion * v_motion +
            w_ai * ai_score +
            scene_bonus
        )
        total_score = _clamp(total_score, 0.0, 1.0)
        scores[i] = total_score

        step_details.append({
            "time": t,
            "score": total_score,
            "audio_wave": a_wave,
            "audio_onset": a_onset,
            "visual_motion": v_motion,
            "ai_score": ai_score,
            "ai_cand": matched_ai,
        })

    return timeline_steps, scores, step_details


def _snap_time_to_boundary(t: float, candidates: List[float], max_delta: float = 0.6) -> float:
    """Snap a timestamp to the closest beat or scene change boundary if within max_delta."""
    if not candidates:
        return t
    best_t = t
    min_dist = max_delta
    for c in candidates:
        dist = abs(c - t)
        if dist < min_dist:
            min_dist = dist
            best_t = c
    return best_t


def _find_top_social_intervals(
    duration: float,
    timeline_steps: np.ndarray,
    scores: np.ndarray,
    clip_count: int = 3,
    duration_mode: str = "auto_15_30",
    beat_times: np.ndarray = None,
    scene_changes: List[float] = None,
) -> List[Dict[str, Any]]:
    """
    Identify the top N non-overlapping highlight intervals that maximize viral engagement.
    """
    if duration <= 5.0:
        return [{
            "start": 0.0,
            "end": duration,
            "duration": duration,
            "metric": 1.0,
            "avg_score": 1.0,
            "peak_score": 1.0,
            "viral_score": 99.9,
            "label": "🔥 Full Video Highlight",
        }]

    # Determine target duration length
    if duration_mode == "15s":
        target_dur = 15.0
        min_dur, max_dur = 13.0, 17.0
    elif duration_mode == "30s":
        target_dur = 30.0
        min_dur, max_dur = 27.0, 33.0
    elif duration_mode == "60s":
        target_dur = 60.0
        min_dur, max_dur = 55.0, 65.0
    else:
        # 'auto_15_30'
        target_dur = min(max(15.0, duration * 0.25), 30.0)
        min_dur, max_dur = 15.0, 30.0

    target_dur = min(target_dur, max(3.0, duration - 0.5))
    min_dur = min(min_dur, target_dur)
    max_dur = min(max_dur, duration)

    snap_points = sorted(set(
        [0.0] +
        ([float(b) for b in beat_times] if beat_times is not None and len(beat_times) > 0 else []) +
        ([float(s) for s in scene_changes] if scene_changes else [])
    ))

    # Evaluate sliding windows across timeline
    window_candidates = []
    step_size = 0.5

    possible_starts = np.arange(0.0, max(0.1, duration - min_dur), step_size)
    for start_t in possible_starts:
        for dur in [target_dur, (min_dur + target_dur) / 2.0, target_dur, (target_dur + max_dur) / 2.0]:
            end_t = min(duration, start_t + dur)
            if end_t - start_t < min_dur:
                continue

            # Snap start and end to beats/scenes
            snapped_start = max(0.0, _snap_time_to_boundary(start_t, snap_points, max_delta=0.6))
            snapped_end = min(duration, _snap_time_to_boundary(end_t, snap_points, max_delta=0.6))
            actual_dur = snapped_end - snapped_start

            if actual_dur < min_dur or actual_dur > max_dur + 2.0:
                continue

            # Compute average score in this interval
            mask = (timeline_steps >= snapped_start) & (timeline_steps <= snapped_end)
            if not np.any(mask):
                continue

            sub_scores = scores[mask]
            avg_score = float(np.mean(sub_scores))
            peak_score = float(np.max(sub_scores))

            # Hook bonus: extra weight if the first 2-3 seconds has high energy
            hook_mask = (timeline_steps >= snapped_start) & (timeline_steps <= min(snapped_end, snapped_start + 3.0))
            hook_score = float(np.mean(scores[hook_mask])) if np.any(hook_mask) else avg_score

            total_metric = 0.5 * avg_score + 0.3 * peak_score + 0.2 * hook_score

            window_candidates.append({
                "start": snapped_start,
                "end": snapped_end,
                "duration": actual_dur,
                "metric": total_metric,
                "avg_score": avg_score,
                "peak_score": peak_score,
            })

    # Sort descending by metric
    window_candidates.sort(key=lambda x: x["metric"], reverse=True)

    # Non-maximum suppression (NMS) to avoid overlapping clips
    selected_intervals = []
    min_separation = max(3.0, min_dur * 0.4)

    for cand in window_candidates:
        c_start, c_end = cand["start"], cand["end"]
        overlap = False
        for sel in selected_intervals:
            s_start, s_end = sel["start"], sel["end"]
            if max(c_start, s_start) < min(c_end, s_end) + min_separation:
                overlap = True
                break
        if not overlap:
            selected_intervals.append(cand)
            if len(selected_intervals) >= clip_count:
                break

    # If we couldn't get enough non-overlapping clips due to short video, relax separation
    if len(selected_intervals) < clip_count and duration >= min_dur * 1.5:
        for cand in window_candidates:
            c_start, c_end = cand["start"], cand["end"]
            overlap = False
            for sel in selected_intervals:
                s_start, s_end = sel["start"], sel["end"]
                intersection = max(0.0, min(c_end, s_end) - max(c_start, s_start))
                if intersection > min(cand["duration"], sel["duration"]) * 0.4:
                    overlap = True
                    break
            if not overlap:
                selected_intervals.append(cand)
                if len(selected_intervals) >= clip_count:
                    break

    # Fallback if no window candidates found
    if not selected_intervals:
        selected_intervals.append({
            "start": 0.0,
            "end": min(duration, target_dur),
            "duration": min(duration, target_dur),
            "metric": 0.8,
            "avg_score": 0.8,
            "peak_score": 0.8,
        })

    # Sort selected intervals chronologically for clean organization
    selected_intervals.sort(key=lambda x: x["start"])

    # Assign labels
    label_templates = [
        "🔥 Climax & Drop Peak",
        "⚡ High-Energy Action",
        "🎬 Dynamic Social Hook",
        "✨ Viral Moment Highlight",
        "🎵 Beat-Drop Climax",
    ]

    for idx, interval in enumerate(selected_intervals):
        scaled_score = round(_clamp(interval["metric"] * 100.0, 10.0, 99.9), 1)
        interval["viral_score"] = scaled_score
        interval["label"] = label_templates[idx % len(label_templates)]

    return selected_intervals


def generate_social_clips(
    video_path: str,
    output_dir: str = None,
    clip_count: int = 3,
    duration_mode: str = "auto_15_30",
    framing_mode: str = "smart_crop",
    ai_strategy: str = "smart_viral",
    enable_qwen_ai: bool = True,
    use_gpu: bool = True,
    gpu_encoder: str = "h264_nvenc",
    custom_fps: float = None,
    enable_subtitles: bool = False,
    lyrics_text: str = None,
    lyrics_mode: str = "auto_whisper",
    lyrics_file: str = None,
    lyrics_style: str = "tiktok_bounce",
    lyrics_palette: str = "tiktok_yellow",
    lyrics_position: str = "bottom",
    progress_callback: Callable[[str], None] = None,
    console_callback: Callable[[int, str], None] = None,
) -> Dict[str, Any]:

    """
    Main function to analyze a full video and export optimal vertical 9:16 clips for TikTok/Shorts.

    Args:
        video_path: Path to input video (newly rendered or external video).
        output_dir: Output folder (defaults to output/tiktok_shorts).
        clip_count: Number of clips to generate (1 to 5).
        duration_mode: 'auto_15_30', '15s', '30s', '60s'.
        framing_mode: 'smart_crop', 'blur_pad', 'fit_letterbox'.
        ai_strategy: 'smart_viral', 'peak_energy_drop', 'visual_action', 'cinematic_beauty'.
        enable_qwen_ai: Toggle local Qwen3-VL analysis.
        use_gpu: Enable NVENC / CUDA acceleration.
        gpu_encoder: 'h264_nvenc', 'hevc_nvenc', or 'cpu'.
        custom_fps: Desired FPS.
        progress_callback: UI progress reporter.
        console_callback: Console stage logger.

    Returns:
        Dict with success status, list of SocialClipResult objects, and metadata.
    """
    start_total_time = time.perf_counter()
    if not video_path or not os.path.exists(video_path):
        return {"success": False, "error": f"Video file not found: {video_path}", "clips": []}

    if output_dir is None:
        output_dir = get_shorts_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    proc_dir = get_processing_dir()
    os.makedirs(proc_dir, exist_ok=True)
    temp_proc_dir = tempfile.mkdtemp(prefix="beatsync_shorts_", dir=proc_dir)


    def _notify(stage_num: int, message: str) -> None:
        if progress_callback:
            progress_callback(f"Stage {stage_num}: {message}")
        if console_callback:
            console_callback(stage_num, message)
        print(f"   📱 [Shorts Clipper] {message}")

    try:
        # Step 1: Video info probe
        _notify(1, "Probing video structure and properties...")
        duration = get_video_duration(video_path)
        fps = custom_fps if custom_fps and custom_fps > 0 else get_video_fps(video_path)
        width, height = get_video_resolution(video_path)
        has_audio = has_audio_stream(video_path)

        _notify(1, f"Source video: {os.path.basename(video_path)} | {width}x{height} @ {fps:.2f} FPS | {duration:.1f}s")

        # Step 2: Audio extraction & rhythm analysis
        audio_info = {}
        extracted_audio = None
        if has_audio:
            _notify(2, "Extracting and analyzing audio wave & beat grid...")
            temp_wav = os.path.join(temp_proc_dir, "audio_analysis.wav")
            if extract_audio_from_video(video_path, temp_wav):
                extracted_audio = temp_wav
                audio_info = _analyze_audio_timeline(temp_wav, duration)
                tempo = audio_info.get("tempo", 120.0)
                beats_count = len(audio_info.get("beat_times", []))
                _notify(2, f"Detected tempo: {tempo:.1f} BPM, {beats_count} rhythmic beats")
            else:
                _notify(2, "Audio extraction failed; continuing with visual-only analysis")
        else:
            _notify(2, "No audio stream detected; using visual motion & AI scoring")

        # Step 3: Visual & Scene detection
        _notify(3, "Analyzing visual scene cuts, motion dynamics, and sharpness...")
        scene_changes = detect_video_scene_changes(video_path, use_gpu=use_gpu)
        keyframes = detect_video_keyframes(video_path)
        all_cuts = sorted(set(scene_changes + keyframes))
        _notify(3, f"Detected {len(scene_changes)} scene cuts, {len(keyframes)} keyframes")

        visual_samples = _measure_visual_motion_sampled(video_path, sample_interval=0.4)
        _notify(3, f"Sampled {len(visual_samples)} visual motion checkpoints")

        # Step 4: AI Semantic Vision (Qwen3-VL)
        ai_candidates = []
        if enable_qwen_ai:
            _notify(4, "Running Qwen3-VL AI Vision Tagging for action, hype & aesthetic moments...")
            ai_candidates = _get_qwen_ai_tags_for_video(video_path, scene_changes, duration, enable_ai=True)
            _notify(4, f"Found {len(ai_candidates)} AI-tagged candidate moments")
        else:
            _notify(4, "Qwen AI vision tagging disabled by user; using deterministic scoring")

        # Step 5: Social Viral Scoring & Highlight Selection
        _notify(5, f"Computing viral engagement curve ({ai_strategy}) and selecting top {clip_count} moments...")
        timeline_steps, scores, step_details = _calculate_social_score_curve(
            duration=duration,
            audio_info=audio_info,
            visual_samples=visual_samples,
            ai_candidates=ai_candidates,
            scene_changes=all_cuts,
            ai_strategy=ai_strategy,
        )

        beat_times_arr = audio_info.get("beat_times", np.array([]))
        intervals = _find_top_social_intervals(
            duration=duration,
            timeline_steps=timeline_steps,
            scores=scores,
            clip_count=clip_count,
            duration_mode=duration_mode,
            beat_times=beat_times_arr,
            scene_changes=all_cuts,
        )

        _notify(5, f"Selected {len(intervals)} optimal social highlight segments")

        # Step 6: Rendering 9:16 Vertical Clips
        _notify(6, f"Rendering {len(intervals)} vertical 9:16 clips ({framing_mode})...")
        effective_use_nvenc = use_gpu and NVENC_AVAILABLE and gpu_encoder != "cpu" and gpu_encoder != "none"
        actual_encoder = gpu_encoder if effective_use_nvenc else "libx264"

        # Prepare full-song lyrics timestamps ONCE across the entire video
        full_timed_words: List[TimedWord] = []
        if enable_subtitles and has_audio and extracted_audio and os.path.exists(extracted_audio):
            _notify(6, "Trascrizione vocale AI e sincronizzazione testi canzone...")
            try:
                from lyrics_karaoke import (
                    transcribe_audio_whisper,
                    align_user_lyrics_with_audio,
                    parse_lrc_or_srt,
                    TimedWord,
                )

                if lyrics_file and os.path.exists(lyrics_file):
                    parsed_phrases = parse_lrc_or_srt(lyrics_file)
                    for p in parsed_phrases:
                        full_timed_words.extend(p.words)
                else:
                    # Transcribe full audio track with Whisper (vad_filter=False captures singing!)
                    whisper_words = transcribe_audio_whisper(
                        audio_path=extracted_audio,
                        model_size="tiny",
                        initial_prompt=lyrics_text,
                        progress_callback=progress_callback,
                    )
                    if lyrics_text and lyrics_text.strip():
                        full_timed_words = align_user_lyrics_with_audio(
                            provided_lyrics_text=lyrics_text,
                            audio_words=whisper_words,
                            audio_duration=duration,
                            beat_times=beat_times_arr,
                        )
                    else:
                        full_timed_words = whisper_words

                _notify(6, f"Rilevate {len(full_timed_words)} parole sincronizzate nella traccia audio.")
            except Exception as e:
                print(f"   ⚠️  Warning preparing subtitles for shorts: {e}")

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        rendered_clips: List[SocialClipResult] = []

        for idx, item in enumerate(intervals):
            start_s = float(item["start"])
            end_s = float(item["end"])
            dur_s = float(item["duration"])
            score_v = float(item["viral_score"])
            lbl = str(item["label"])

            clip_filename = f"{base_name}_short_{idx+1}_{_fmt_time_badge(start_s).replace(':','m')}s_{timestamp}_9x16.mp4"
            clip_output_path = os.path.join(output_dir, clip_filename)

            # Generate clip-specific ASS subtitle file if enabled
            clip_ass_file = None
            if enable_subtitles and (full_timed_words or (lyrics_text and lyrics_text.strip())):
                try:
                    from lyrics_karaoke import (
                        group_words_into_phrases,
                        generate_karaoke_ass,
                        align_user_lyrics_with_audio,
                        TimedWord,
                    )

                    clip_words = []
                    if full_timed_words:
                        for tw in full_timed_words:
                            if tw.end > start_s and tw.start < end_s:
                                w_s = max(0.0, tw.start - start_s)
                                w_e = min(dur_s, tw.end - start_s)
                                if w_e > w_s:
                                    clip_words.append(TimedWord(
                                        word=tw.word,
                                        start=w_s,
                                        end=w_e,
                                        confidence=tw.confidence,
                                    ))

                    # If this specific clip fell on an instrumental section with no vocals, but user gave lyrics:
                    if not clip_words and lyrics_text and lyrics_text.strip():
                        clip_beats = [float(b) - start_s for b in beat_times_arr if start_s <= float(b) <= end_s]
                        clip_words = align_user_lyrics_with_audio(
                            provided_lyrics_text=lyrics_text,
                            audio_duration=dur_s,
                            beat_times=clip_beats if len(clip_beats) >= 2 else None,
                        )

                    if clip_words:
                        clip_phrases = group_words_into_phrases(clip_words, max_words_per_line=3)
                        if clip_phrases:
                            clip_ass_file = os.path.join(temp_proc_dir, f"clip_{idx+1}_karaoke.ass")
                            generate_karaoke_ass(
                                phrases=clip_phrases,
                                output_ass_path=clip_ass_file,
                                video_width=DEFAULT_TARGET_WIDTH,
                                video_height=DEFAULT_TARGET_HEIGHT,
                                animation_style=lyrics_style,
                                palette_key=lyrics_palette,
                                position_mode=lyrics_position,
                                uppercase=True,
                            )
                            _notify(6, f"Clip #{idx+1}: Creati sottotitoli animati ({len(clip_words)} parole)")
                except Exception as sub_e:
                    print(f"   ⚠️  Warning generating clip subtitle: {sub_e}")

            _notify(6, f"Extracting Clip {idx+1}/{len(intervals)}: {start_s:.2f}s - {end_s:.2f}s ({dur_s:.1f}s) -> 1080x1920...")




            success, err_msg = extract_vertical_clip_9_16(
                video_file=video_path,
                start_time=start_s,
                duration=dur_s,
                output_file=clip_output_path,
                framing_mode=framing_mode,
                target_width=DEFAULT_TARGET_WIDTH,
                target_height=DEFAULT_TARGET_HEIGHT,
                fps=fps,
                use_nvenc=effective_use_nvenc,
                gpu_encoder=gpu_encoder if effective_use_nvenc else "h264_nvenc",
                audio_fade=True,
                ass_subtitle_file=clip_ass_file,
            )


            if not success or not os.path.exists(clip_output_path) or os.path.getsize(clip_output_path) == 0:
                print(f"   ⚠️  Warning: Failed to render clip {idx+1}: {err_msg}")
                continue

            clip_res = SocialClipResult(
                clip_index=idx + 1,
                file_path=clip_output_path,
                start_time=start_s,
                end_time=end_s,
                duration=dur_s,
                viral_score=score_v,
                label=lbl,
                framing_mode=framing_mode,
                ai_tags={"strategy": ai_strategy, "encoder": actual_encoder},
            )
            rendered_clips.append(clip_res)

        total_elapsed = time.perf_counter() - start_total_time
        _notify(6, f"Successfully created {len(rendered_clips)} vertical 9:16 clips in {_fmt_seconds(total_elapsed)}!")

        return {
            "success": True,
            "clips": rendered_clips,
            "clip_count": len(rendered_clips),
            "output_dir": output_dir,
            "total_processing_seconds": total_elapsed,
            "source_duration": duration,
            "framing_mode": framing_mode,
            "ai_strategy": ai_strategy,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "clips": [],
        }
    finally:
        try:
            if os.path.exists(temp_proc_dir):
                shutil.rmtree(temp_proc_dir, ignore_errors=True)
        except Exception:
            pass
