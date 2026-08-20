import os
import sys
import contextlib
import asyncio

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def _install_windows_asyncio_connection_reset_filter() -> None:
    """Hide benign Windows asyncio pipe resets after browser/subprocess shutdown.

    On Windows, asyncio's Proactor transport can log a scary traceback when a
    local socket or subprocess pipe is closed by the other side after the real
    work is already complete. The app result is not affected, so suppress only
    that exact WinError 10054 callback and let all other async errors through.
    """
    if os.name != "nt" or getattr(asyncio, "_beatsync_win10054_filter", False):
        return
    asyncio._beatsync_win10054_filter = True

    def is_benign_reset(exc: BaseException | None) -> bool:
        if not isinstance(exc, ConnectionResetError):
            return False
        winerror = getattr(exc, "winerror", None)
        errno_value = getattr(exc, "errno", None)
        return winerror == 10054 or errno_value == 10054 or "WinError 10054" in str(exc)

    # Directly patch the noisy Proactor pipe cleanup callback when available.
    try:
        from asyncio import proactor_events

        transport_cls = getattr(proactor_events, "_ProactorBasePipeTransport", None)
        original_call_lost = getattr(transport_cls, "_call_connection_lost", None)
        if transport_cls is not None and original_call_lost is not None:
            def quiet_call_connection_lost(self, exc):  # type: ignore[no-untyped-def]
                try:
                    return original_call_lost(self, exc)
                except ConnectionResetError as reset_exc:
                    if is_benign_reset(reset_exc):
                        return None
                    raise

            transport_cls._call_connection_lost = quiet_call_connection_lost
    except Exception:
        pass

    # Fallback for the same exception if it still reaches the loop logger.
    original_exception_handler = asyncio.BaseEventLoop.call_exception_handler

    def quiet_exception_handler(self, context):  # type: ignore[no-untyped-def]
        exc = context.get("exception") if isinstance(context, dict) else None
        handle = str(context.get("handle", "")) if isinstance(context, dict) else ""
        if is_benign_reset(exc) and "_ProactorBasePipeTransport._call_connection_lost" in handle:
            return None
        return original_exception_handler(self, context)

    asyncio.BaseEventLoop.call_exception_handler = quiet_exception_handler


_install_windows_asyncio_connection_reset_filter()

from logger import (
    setup_environment,
    USING_PORTABLE_PYTHON, USING_PORTABLE_CUDA, USING_CUPY_CTK, FFMPEG_FOUND
)

# Initialize environment
setup_environment()
# NOW import other modules (after CUDA environment is set)
import gradio as gr
import tempfile
import shutil
import datetime
import multiprocessing
import queue
import re
import subprocess
import threading
import time
import socket
from typing import Callable, Iterator, TypeAlias, Tuple, Dict, List, Any

# Import FFmpeg processing module
from ffmpeg_processing import get_video_fps, FFMPEG_PATH

# Shared runtime settings
from gpu_cpu_utils import (
    CPU_COUNT,
    MAX_THREADS,
    PARALLEL_WORKERS,
    GPU_INFO,
    GPU_AVAILABLE,
    NVENC_AVAILABLE,
    set_gpu_mode,
)

from paths import (
    GRADIO_TEMP_DIR,
    get_input_dir,
    get_audio_input_dir,
    get_video_input_dir,
    get_output_dir,
    get_processing_dir,
    get_shorts_output_dir,
    get_subtitles_output_dir,
)


gpu_data = GPU_INFO
gpu_info = f"{gpu_data['name']} ({gpu_data['cuda_version']})" if gpu_data['available'] else "CPU Mode"

from video_processor import create_music_video
from auto_mode import analyze_beats_auto
from social_clipper import generate_social_clips

# Import UI content
from ui_content import *

# Set environment variable for Gradio
os.environ['GRADIO_TEMP_DIR'] = GRADIO_TEMP_DIR

VideoFilesInput : TypeAlias = List[str]
StatusResult : TypeAlias = Tuple[str, str, Dict]

STATUS_BOX_CSS = """
#status-output-box, #shorts-status-box {
    min-height: 120px !important;
}

#status-output-box textarea, #shorts-status-box textarea {
    height: 110px !important;
    min-height: 110px !important;
    max-height: 160px !important;
    overflow-y: auto !important;
    resize: none !important;
    font-family: monospace;
    font-size: 0.92rem;
}

.shorts-video-player {
    max-height: 480px !important;
}
"""


def _stage_status(stage_number: int) -> str:
    return f"Stage {stage_number} is processing. Please wait."



class QuietConsole:
    """Discard legacy verbose prints while the Gradio worker runs."""

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass


class StageConsoleLogger:
    """Small CMD logger: stage start, up to 5 useful lines, stage end."""

    def __init__(self, stream, max_lines_per_stage: int = 5):
        self.stream = stream
        self.max_lines_per_stage = max(1, int(max_lines_per_stage))
        self.stage_number: int | None = None
        self.stage_started = 0.0
        self.stage_line_count = 0
        self.total_started = time.perf_counter()

    def start_stage(self, stage_number: int) -> None:
        if self.stage_number == stage_number:
            return
        self.end_stage()
        self.stage_number = stage_number
        self.stage_started = time.perf_counter()
        self.stage_line_count = 0
        self._write(f"Stage {stage_number} processing started:\n")

    def stage_line(self, stage_number: int, message: str) -> None:
        if self.stage_number != stage_number:
            self.start_stage(stage_number)
        self.line(message)

    def line(self, message: str) -> None:
        if self.stage_number is None:
            return
        if self.stage_line_count >= self.max_lines_per_stage:
            return
        message = self._clean(message)
        if message:
            self._write(f"  {message}\n")
            self.stage_line_count += 1

    def end_stage(self) -> None:
        if self.stage_number is None:
            return
        elapsed = int(round(time.perf_counter() - self.stage_started))
        self._write(f"Stage {self.stage_number} ended in {elapsed} seconds.\n\n")
        self.stage_number = None
        self.stage_started = 0.0
        self.stage_line_count = 0

    def finish(self) -> None:
        self.end_stage()
        elapsed = int(round(time.perf_counter() - self.total_started))
        self._write(f"Total time processing: {elapsed} seconds\n")

    def _write(self, text: str) -> None:
        self.stream.write(text)
        self.stream.flush()

    def _clean(self, text: str) -> str:
        text = str(text).encode("ascii", "ignore").decode("ascii")
        return re.sub(r"\s+", " ", text).strip()


def _fmt_stage_seconds(seconds: float | int | None) -> str:
    try:
        return f"{float(seconds):.1f}s"
    except Exception:
        return "0.0s"


def _short_model_name(model_id: str | None) -> str:
    if not model_id:
        return ""
    return os.path.basename(str(model_id).rstrip("/\\")) or str(model_id)


def _stage5_summary(console_logger: StageConsoleLogger | None, video_analysis: Dict | None) -> None:
    if console_logger is None or not isinstance(video_analysis, dict):
        return

    source_count = int(video_analysis.get("source_count") or len(video_analysis.get("videos") or []))
    worker_count = int(video_analysis.get("worker_count") or 1)
    cache_hits = int(video_analysis.get("cache_hits") or 0)
    ai_enabled = bool(video_analysis.get("ai_enabled"))
    qwen_total = int(video_analysis.get("qwen_frame_count") or 0)
    qwen_tags = int(video_analysis.get("qwen_tag_count") or 0)
    model_id = _short_model_name(video_analysis.get("qwen_model_id"))
    batch_size = int(video_analysis.get("qwen_concurrency") or 0)

    console_logger.line(f"Source videos: {source_count}, visual workers: {worker_count}")
    if ai_enabled:
        qwen_bits = ["Qwen: enabled"]
        if model_id:
            qwen_bits.append(f"model {model_id}")
        if batch_size:
            qwen_bits.append(f"batch {batch_size}")
        console_logger.line(", ".join(qwen_bits))
        if qwen_total:
            inference_seconds = float(video_analysis.get("qwen_inference_seconds") or video_analysis.get("qwen_seconds") or 0.0)
            qwen_rate = (qwen_total / inference_seconds) if inference_seconds > 0 else 0.0
            peak_vram = float(video_analysis.get("qwen_peak_vram_gb") or 0.0)
            perf_bits = []
            if batch_size:
                perf_bits.append(f"batch {batch_size}")
            if peak_vram > 0:
                perf_bits.append(f"~{peak_vram:.2f} GB VRAM")
            if qwen_rate > 0:
                perf_bits.append(f"{qwen_rate:.2f} candidates/s")
            if perf_bits:
                console_logger.line(f"Qwen performance: {', '.join(perf_bits)}")
            console_logger.line(
                f"Qwen tags: {qwen_tags}/{qwen_total} in {_fmt_stage_seconds(video_analysis.get('qwen_seconds'))}"
            )
    else:
        console_logger.line("Qwen: disabled")

    summary = video_analysis.get("summary")
    if summary:
        console_logger.line(f"Visual library: {summary}")
    console_logger.line(
        f"Analysis time: {_fmt_stage_seconds(video_analysis.get('analysis_seconds'))}, cache {cache_hits}/{source_count}"
    )


def _stage6_summary(console_logger: StageConsoleLogger | None, beat_info: Dict | None) -> None:
    if console_logger is None or not isinstance(beat_info, dict):
        return

    render_info = beat_info.get("render_info") or {}
    if not render_info:
        return

    cuts = int(render_info.get("render_cuts") or 0)
    frames = int(render_info.get("timeline_frames") or 0)
    fps = render_info.get("output_fps")
    if cuts or frames:
        fps_text = f" @ {float(fps):.1f} FPS" if fps is not None else ""
        console_logger.line(f"Render timeline: {cuts} cuts, {frames} frames{fps_text}")

    clip_workers = render_info.get("clip_workers")
    requested_workers = render_info.get("requested_workers")
    worker_text = ""
    if clip_workers:
        worker_text = f", workers {clip_workers}"
        if requested_workers and requested_workers != clip_workers:
            worker_text += f"/{requested_workers}"
    encoder = render_info.get("encoder")
    if encoder or worker_text:
        console_logger.line(f"Encoder: {encoder or 'unknown'}{worker_text}")

    if render_info.get("audio_duration") is not None:
        console_logger.line(f"Audio duration: {float(render_info['audio_duration']):.2f} seconds")

    plan_summary = render_info.get("plan_summary") or {}
    if plan_summary:
        console_logger.line(
            "Planner: "
            f"{int(plan_summary.get('clip_count') or 0)} clips, "
            f"{int(plan_summary.get('source_count') or 0)} sources, "
            f"AI moments {int(plan_summary.get('ai_tagged') or 0)}"
        )

    final_bits = []
    if render_info.get("target_resolution"):
        final_bits.append(f"resolution {render_info['target_resolution']}")
    if render_info.get("final_assembly_seconds") is not None:
        final_bits.append(f"assembly {_fmt_stage_seconds(render_info['final_assembly_seconds'])}")
    if final_bits:
        console_logger.line("Final: " + ", ".join(final_bits))

def find_launch_port(default_port: int = 7860, search_limit: int = 20) -> int:
    """Prefer the default Gradio port, then step forward if it is busy."""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            print(f"⚠️ Invalid GRADIO_SERVER_PORT={env_port!r}; using auto port search.")

    for port in range(default_port, default_port + search_limit):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                if port != default_port:
                    print(f"⚠️ Port {default_port} is busy. Starting BeatSync on port {port}.")
                return port

    raise OSError(f"Cannot find empty port in range: {default_port}-{default_port + search_limit - 1}")


def _as_existing_source_path(file_path: str | None) -> str | None:
    """Use the selected source file directly instead of copying it locally."""
    if not file_path:
        return None
    try:
        path = os.path.abspath(os.fspath(file_path))
    except TypeError:
        return None
    return path if os.path.isfile(path) else None


def _as_existing_source_paths(file_paths: VideoFilesInput) -> list[str]:
    if not file_paths:
        return []
    return [path for path in (_as_existing_source_path(p) for p in file_paths) if path]


def _process_video_impl(audio_file: str, video_files: VideoFilesInput,
                       output_filename: str, processing_mode: str,
                       custom_fps: float,
                       enable_subtitles: bool = False,
                       lyrics_mode: str = "auto_whisper",
                       lyrics_text: str = None,
                       lyrics_file: str = None,
                       lyrics_style: str = "tiktok_bounce",
                       lyrics_palette: str = "tiktok_yellow",
                       lyrics_position: str = "bottom",
                       session_state: dict = None,
                       progress_callback: Callable[[str], None] | None = None,
                       console_logger: StageConsoleLogger | None = None) -> StatusResult:
    total_started = time.perf_counter()
    parallel_workers = PARALLEL_WORKERS
    session_dir = tempfile.mkdtemp(prefix='beatsync_session_')
    session_state = dict(session_state or {})
    
    try:
        # Prepare AI Vision model if selected

        qwen_enabled = True
        qwen_model_path = ""
        if ai_vision_tier in ("standard_2b", "pro_7b"):
            from model_manager import ensure_model_available, get_model_paths
            ensure_model_available(ai_vision_tier, progress_callback)
            m_paths = get_model_paths(ai_vision_tier)
            if m_paths.get("model"):
                os.environ["BEATSYNC_QWEN_LLAMA_MODEL"] = m_paths["model"]
            if m_paths.get("mmproj"):
                os.environ["BEATSYNC_QWEN_LLAMA_MMPROJ"] = m_paths["mmproj"]
            qwen_model_path = m_paths.get("model", "")
            qwen_enabled = True
        elif ai_vision_tier == "fast_siglip":
            qwen_model_path = ""
            qwen_enabled = False

        if progress_callback:
            progress_callback(_stage_status(1))
        
        # Handle audio
        if audio_file:
            local_audio_path = _as_existing_source_path(audio_file)
            if local_audio_path:
                session_state['local_audio_path'] = local_audio_path
                session_state['original_audio_path'] = audio_file
            else:
                return None, '❌ Error: Could not access audio file', session_state
        elif session_state.get('local_audio_path'):
            local_audio_path = session_state.get('local_audio_path')
        else:
            return None, '❌ Error: No audio file selected', session_state

        # Handle videos
        if video_files:
            if video_files != session_state.get('original_video_paths'):
                local_video_paths = _as_existing_source_paths(video_files)
                if local_video_paths:
                    session_state['local_video_paths'] = local_video_paths
                    session_state['original_video_paths'] = video_files
                else:
                    return None, '❌ Error: Could not access video files', session_state
            else:
                local_video_paths = session_state.get('local_video_paths')
        else:
            return None, '❌ Error: No video files selected', session_state

        # Verify files exist
        if not local_audio_path or not os.path.exists(local_audio_path):
             return None, f"❌ Error: Audio file is missing or inaccessible.", session_state
        if not local_video_paths or not all(p and os.path.exists(p) for p in local_video_paths):
             return None, f"❌ Error: Video files are missing or inaccessible.", session_state
        
        # Set GPU mode
        use_gpu = GPU_AVAILABLE
        set_gpu_mode(use_gpu)
        
        # Determine processing mode
        is_prores = processing_mode == 'prores_proxy'
        use_nvenc = (processing_mode in ['h264_nvenc', 'hevc_nvenc']) and NVENC_AVAILABLE
        gpu_encoder = processing_mode if use_nvenc else 'none'
        
        python_str = "Portable" if USING_PORTABLE_PYTHON else "System"
        cuda_str = "CuPy CTK" if USING_CUPY_CTK else ("Portable" if USING_PORTABLE_CUDA else "System/None")

        # Determine FPS
        if custom_fps is not None and custom_fps > 0:
            output_fps = custom_fps
        else:
            output_fps = get_video_fps(local_video_paths[0])
            
        # Prepare output paths
        output_folder = get_output_dir()
        os.makedirs(output_folder, exist_ok=True)
        name, _ = os.path.splitext(output_filename)
        ext = '.mov' if is_prores else '.mp4'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}{ext}"
        output_path = os.path.join(output_folder, filename)
        temp_output = os.path.join(session_dir, filename)

        selected_beats, beat_info = analyze_beats_auto(
            local_audio_path,
            use_gpu=use_gpu,
            video_files=local_video_paths,
            enable_qwen_semantics=qwen_enabled,
            qwen_model_path=qwen_model_path,
            progress_callback=progress_callback,
            console_callback=lambda stage, message: console_logger.stage_line(stage, message) if console_logger else None,
        )
        beat_times = beat_info.get('times', selected_beats)
        _stage5_summary(console_logger, beat_info.get("video_analysis"))

        if progress_callback:
            progress_callback(_stage_status(6))

        # Create video
        result_path = create_music_video(
            local_audio_path, local_video_paths, selected_beats,
            output_file=temp_output, max_workers=parallel_workers,
            beat_info=beat_info, lossless_mode=is_prores,
            use_gpu=use_gpu, gpu_encoder=gpu_encoder, fps=output_fps,
            enable_subtitles=bool(enable_subtitles),
            lyrics_text=lyrics_text,
            lyrics_mode=lyrics_mode,
            lyrics_file=lyrics_file,
            lyrics_style=lyrics_style,
            lyrics_palette=lyrics_palette,
            lyrics_position=lyrics_position,
            progress_callback=progress_callback,
        )

        # Move to output folder
        shutil.move(result_path, output_path)

        # Create preview for ProRes if needed
        preview_path = output_path
        if is_prores:
            preview_filename = f"{name}_{timestamp}_preview.mp4"
            preview_path = os.path.join(session_dir, preview_filename)
            preview_cmd = [FFMPEG_PATH]
            if NVENC_AVAILABLE:
                preview_cmd.extend(['-hwaccel', 'cuda', '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '23'])
            else:
                preview_cmd.extend(['-hwaccel', 'auto', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23'])
            preview_cmd.extend(['-i', output_path, '-pix_fmt', 'yuv420p', '-y', preview_path])
            subprocess.run(preview_cmd, capture_output=True, text=True, timeout=180)
        _stage6_summary(console_logger, beat_info)

        # Generate status message based on mode
        gpu_info = f"⚡ GPU: {GPU_INFO}" if use_gpu else "💻 CPU"
        fps_info = f"{output_fps:.2f} FPS (custom)" if custom_fps else f"{output_fps:.2f} FPS (auto-detected)"
        audio_info = "PCM 24-bit (48kHz)" if is_prores else "AAC 320kbps (48kHz)"
        
        if is_prores:
            codec_info = "ProRes 422 Proxy (.mov) - Lossless"
            encoder_info = "🎯 Lossless Concatenation"
        elif use_nvenc:
            codec_info = f"{gpu_encoder.upper()} (.mp4)"
            encoder_info = f"⚡ {gpu_encoder.upper()}"
        else:
            codec_info = "H.264 (.mp4)"
            encoder_info = "💻 libx264"

        total_cuts = len(selected_beats) - 1
        sections_info = beat_info.get('selection_info', [])
        total_processing_seconds = time.perf_counter() - total_started
        processing_label = gpu_encoder.upper() if use_nvenc else ("PRORES_PROXY" if is_prores else "H264_CPU")
        
        status_msg = get_success_message_auto(
            total_cuts, len(beat_times),
            beat_info.get('tempo', 120), sections_info,
            python_str, cuda_str, MAX_THREADS, CPU_COUNT,
            parallel_workers, gpu_info, encoder_info,
            codec_info, fps_info, filename, audio_info,
            audio_duration=beat_info.get('audio_duration'),
            output_fps=output_fps,
            total_processing_seconds=total_processing_seconds,
            processing_label=processing_label
        )
        # Return preview path for display, keep session_state intact
        return preview_path, status_msg, session_state

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        import traceback
        traceback.print_exc()
        return None, error_msg, session_state


def process_video(
    audio_file: str,
    video_files: List[str],
    output_filename: str,
    processing_mode: str,
    custom_fps: float,
    ai_vision_tier: str,
    enable_subtitles: bool,
    lyrics_mode: str,
    lyrics_text: str,
    lyrics_file: str,
    lyrics_style: str,
    lyrics_palette: str,
    lyrics_position: str,
    session_state: dict
) -> Iterator[StatusResult]:
    status_queue: queue.Queue[str | None] = queue.Queue()
    result_queue: queue.Queue[StatusResult] = queue.Queue(maxsize=1)
    console_logger = StageConsoleLogger(sys.__stdout__)
    quiet_console = QuietConsole()

    def progress_callback(msg: str) -> None:
        status_queue.put(msg)
        match = re.search(r"Stage (\d+):", msg)
        if match:
            console_logger.start_stage(int(match.group(1)))

    def worker() -> None:
        try:
            with contextlib.redirect_stdout(quiet_console), contextlib.redirect_stderr(quiet_console):
                res = _process_video_impl(
                    audio_file=audio_file,
                    video_files=video_files,
                    output_filename=output_filename,
                    processing_mode=processing_mode,
                    custom_fps=custom_fps,
                    ai_vision_tier=ai_vision_tier,
                    enable_subtitles=enable_subtitles,
                    lyrics_mode=lyrics_mode,
                    lyrics_text=lyrics_text,
                    lyrics_file=lyrics_file,
                    lyrics_style=lyrics_style,
                    lyrics_palette=lyrics_palette,
                    lyrics_position=lyrics_position,
                    session_state=session_state,
                    progress_callback=progress_callback,
                    console_logger=console_logger
                )
        except Exception as e:
            console_logger.line(f"Error: {e}")
            res = None, f"❌ Error: {e}", session_state
        finally:
            console_logger.finish()
        result_queue.put(res)
        status_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    console_logger.start_stage(1)
    thread.start()

    last_status = _stage_status(1)
    yield None, last_status, session_state

    while True:
        message = status_queue.get()
        if message is None:
            break
        if message != last_status:
            last_status = message
            yield None, message, session_state

    thread.join()
    yield result_queue.get()


# ============================================================================
# TIKTOK / SHORTS PROCESSING HANDLERS
# ============================================================================

def _format_shorts_summary_markdown(result: Dict[str, Any]) -> str:
    if not result.get("success"):
        return f"❌ **Errore generazione clip**: {result.get('error', 'Sconosciuto')}"

    clips = result.get("clips", [])
    if not clips:
        return "⚠️ Nessuna clip estratta."

    total_s = result.get("total_processing_seconds", 0.0)
    lines = [
        f"### ✅ {len(clips)} Clip TikTok / Shorts (9:16) Generate con Successo!",
        f"⏱️ **Tempo di elaborazione**: {total_s:.1f}s | 📐 **Formato**: 1080x1920 (9:16) | 🧠 **Strategia**: `{result.get('ai_strategy')}`",
        "",
        "| Clip | Titolo / Highlight | Range Temporale | Durata | Punteggio Virale | Nome File |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for c in clips:
        st_badge = f"{int(c.start_time//60):02d}:{c.start_time%60:04.1f}"
        et_badge = f"{int(c.end_time//60):02d}:{c.end_time%60:04.1f}"
        lines.append(
            f"| **#{c.clip_index}** | {c.label} | `{st_badge} - {et_badge}` | `{c.duration:.1f}s` | **{c.viral_score}/100** 🔥 | `{c.filename}` |"
        )

    lines.append("")
    lines.append(f"📁 *Cartella di output:* `{result.get('output_dir')}`")
    return "\n".join(lines)


def _process_social_clips_impl(
    video_file: str,
    clip_count: int,
    duration_mode: str,
    framing_mode: str,
    ai_strategy: str,
    enable_qwen: bool,
    ai_vision_tier: str,
    processing_mode: str,
    enable_subtitles: bool,
    lyrics_mode: str,
    lyrics_text: str,
    lyrics_file: str,
    lyrics_style: str,
    lyrics_palette: str,
    lyrics_position: str,
    progress_callback: Callable[[str], None] = None,
    console_logger: StageConsoleLogger | None = None,
) -> Tuple[str, List[str | None], str]:
    if not video_file:
        return "❌ Error: No video file selected.", [None] * 5, ""

    local_path = _as_existing_source_path(video_file)
    if not local_path or not os.path.exists(local_path):
        return f"❌ Error: Video file not accessible: {video_file}", [None] * 5, ""

    use_gpu = GPU_AVAILABLE
    set_gpu_mode(use_gpu)
    use_nvenc = (processing_mode in ['h264_nvenc', 'hevc_nvenc']) and NVENC_AVAILABLE
    gpu_encoder = processing_mode if use_nvenc else 'cpu'

    if progress_callback:
        progress_callback("Starting audio-visual & AI analysis of video...")

    res = generate_social_clips(
        video_path=local_path,
        clip_count=int(clip_count),
        duration_mode=duration_mode,
        framing_mode=framing_mode,
        ai_strategy=ai_strategy,
        enable_qwen_ai=bool(enable_qwen),
        ai_vision_tier=ai_vision_tier,
        use_gpu=use_gpu,
        gpu_encoder=gpu_encoder,
        enable_subtitles=bool(enable_subtitles),
        lyrics_text=lyrics_text,
        lyrics_mode=lyrics_mode,
        lyrics_file=lyrics_file,
        lyrics_style=lyrics_style,
        lyrics_palette=lyrics_palette,
        lyrics_position=lyrics_position,
        progress_callback=progress_callback,
        console_callback=lambda stage, msg: console_logger.stage_line(stage, msg) if console_logger else None,
    )

    if not res.get("success"):
        return f"❌ Error: {res.get('error', 'Generation failed')}", [None] * 5, ""

    clips = res.get("clips", [])
    clip_paths = [None] * 5
    for idx, c in enumerate(clips[:5]):
        clip_paths[idx] = c.file_path

    summary_md = _format_shorts_summary_markdown(res)
    status_text = f"✅ Completed! {len(clips)} vertical 9:16 clips created in {res.get('total_processing_seconds', 0):.1f}s."
    return status_text, clip_paths, summary_md


def process_social_clips(
    video_file: str,
    clip_count: int,
    duration_mode: str,
    framing_mode: str,
    ai_strategy: str,
    enable_qwen: bool,
    ai_vision_tier: str,
    processing_mode: str,
    enable_subtitles: bool,
    lyrics_mode: str,
    lyrics_text: str,
    lyrics_file: str,
    lyrics_style: str,
    lyrics_palette: str,
    lyrics_position: str,
) -> Iterator[Tuple[str, str | None, str | None, str | None, str | None, str | None, str]]:
    status_queue: queue.Queue[str | None] = queue.Queue()
    result_queue: queue.Queue[Tuple[str, List[str | None], str]] = queue.Queue(maxsize=1)
    console_logger = StageConsoleLogger(sys.__stdout__)
    quiet_console = QuietConsole()

    def progress_callback(msg: str) -> None:
        status_queue.put(msg)
        match = re.search(r"Stage (\d+):", msg)
        if match:
            console_logger.start_stage(int(match.group(1)))

    def worker() -> None:
        try:
            with contextlib.redirect_stdout(quiet_console), contextlib.redirect_stderr(quiet_console):
                res = _process_social_clips_impl(
                    video_file=video_file,
                    clip_count=clip_count,
                    duration_mode=duration_mode,
                    framing_mode=framing_mode,
                    ai_strategy=ai_strategy,
                    enable_qwen=enable_qwen,
                    ai_vision_tier=ai_vision_tier,
                    processing_mode=processing_mode,
                    enable_subtitles=enable_subtitles,
                    lyrics_mode=lyrics_mode,
                    lyrics_text=lyrics_text,
                    lyrics_file=lyrics_file,
                    lyrics_style=lyrics_style,
                    lyrics_palette=lyrics_palette,
                    lyrics_position=lyrics_position,
                    progress_callback=progress_callback,
                    console_logger=console_logger,
                )
        except Exception as e:
            console_logger.line(f"Error: {e}")
            res = f"❌ Error: {e}", [None] * 5, ""
        finally:
            console_logger.finish()
        result_queue.put(res)
        status_queue.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    console_logger.start_stage(1)
    thread.start()

    last_status = "Inizializzazione estrazione clip 9:16..."
    yield last_status, None, None, None, None, None, ""

    while True:
        msg = status_queue.get()
        if msg is None:
            break
        if msg != last_status:
            last_status = msg
            yield msg, None, None, None, None, None, ""

    thread.join()
    final_status, clip_paths, summary_md = result_queue.get()
    yield final_status, clip_paths[0], clip_paths[1], clip_paths[2], clip_paths[3], clip_paths[4], summary_md


def cleanup_on_startup():
    """
    Clean temporary runtime files on script start while preserving user inputs
    and the persistent video analysis cache.
    """
    input_base = get_input_dir()
    protected_dirs = {'audio', 'video', 'video_analysis_cache'}

    try:
        os.makedirs(get_audio_input_dir(), exist_ok=True)
        os.makedirs(get_video_input_dir(), exist_ok=True)
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)

        if os.path.exists(input_base):
            for item in os.listdir(input_base):
                item_path = os.path.join(input_base, item)

                # Keep the latest user input files across restarts.
                if item in protected_dirs:
                    continue

                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                except Exception as e:
                    print(f"   ⚠️  Could not clean {item}: {e}")

        # Recreate runtime temp upload and processing folder after cleanup.
        os.makedirs(os.path.join(input_base, 'gradio_uploads'), exist_ok=True)
        os.makedirs(get_processing_dir(), exist_ok=True)
        os.makedirs(get_shorts_output_dir(), exist_ok=True)

    except Exception as e:
        print(f"   ⚠️  Warning during startup cleanup: {e}")



def create_ui() -> gr.Blocks:
    python_status = "✅ Portable (bin/python-3.13.14-embed-amd64/)" if USING_PORTABLE_PYTHON else "⚠️  System Python"
    if USING_CUPY_CTK:
        cuda_status = "✅ CuPy CTK (Python wheel libraries)"
    elif USING_PORTABLE_CUDA:
        cuda_status = "✅ Portable (bin/CUDA/v13.3)"
    else:
        cuda_status = "⚠️  System CUDA (or not available)"
    ffmpeg_status = "✅ Portable (bin/ffmpeg/)" if FFMPEG_FOUND else "⚠️  System FFmpeg"
    
    app = gr.Blocks(title='BeatSync Engine', theme='ocean', css=STATUS_BOX_CSS)
    with app:
        session_state = gr.State({})

        gr.Markdown(f"# {UI_TITLE}")
        gr.Markdown(UI_MAIN_DESCRIPTION)
        
        with gr.Tabs() as main_tabs:
            # ============================================================================
            # TAB 1: FULL MUSIC VIDEO GENERATOR
            # ============================================================================
            with gr.Tab(TAB_FULL_VIDEO, id="full_video_tab"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('### 📁 Input Files')
                        audio_input = gr.File(label=LABEL_AUDIO_FILE, file_types=['.mp3', '.wav', '.flac'], type='filepath', elem_id='audio-file-input')
                        video_input = gr.File(label=LABEL_VIDEO_FILES, file_count='multiple', file_types=['.mp4', '.mkv'], type='filepath', elem_id='video-files-input')

                        with gr.Group():
                            gr.Markdown('### ⚙️ Video Settings')
                            custom_fps = gr.Number(label=LABEL_CUSTOM_FPS, value=None, precision=2, info=INFO_CUSTOM_FPS)

                        with gr.Group():
                            gr.Markdown(f'### 🎬 Processing Mode')
                            if NVENC_AVAILABLE:
                                processing_mode = gr.Radio(choices=[('NVIDIA NVENC H.264', 'h264_nvenc'), ('NVIDIA NVENC HEVC (H.265)', 'hevc_nvenc'), ('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='h264_nvenc', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_nvenc())
                            else:
                                processing_mode = gr.Radio(choices=[('CPU (H.264)', 'cpu'), ('ProRes 422 Proxy (Precise Mode)', 'prores_proxy')], value='cpu', label=LABEL_PROCESSING_MODE, info=get_processing_mode_info_cpu())
                        
                        with gr.Group():
                            gr.Markdown('### 📁 Output Settings')
                            output_filename = gr.Textbox(value='music_video.mp4', label=LABEL_OUTPUT_FILENAME, info=INFO_OUTPUT_FILENAME)

                        with gr.Group():
                            gr.Markdown('### 🧠 AI Vision Intelligence')
                            mv_ai_vision_tier = gr.Dropdown(
                                choices=CHOICES_AI_VISION_TIER,
                                value="standard_2b",
                                label=LABEL_AI_VISION_TIER,
                                info=INFO_AI_VISION_TIER,
                            )

                        with gr.Accordion(GROUP_KARAOKE_TITLE, open=False):
                            enable_subtitles = gr.Checkbox(label=LABEL_ENABLE_KARAOKE, value=False, info=INFO_ENABLE_KARAOKE)
                            with gr.Row():
                                lyrics_mode = gr.Radio(choices=CHOICES_LYRICS_MODE, value="auto_whisper", label=LABEL_LYRICS_MODE)
                                lyrics_file = gr.File(label=LABEL_LYRICS_FILE, file_types=['.lrc', '.srt', '.ass', '.txt'], type='filepath')
                            lyrics_text = gr.Textbox(label=LABEL_LYRICS_TEXT, placeholder="Paste full song lyrics here to synchronize with beats and vocals...", lines=3, info=INFO_LYRICS_TEXT)
                            with gr.Row():
                                lyrics_style = gr.Dropdown(choices=CHOICES_LYRICS_STYLE, value="tiktok_bounce", label=LABEL_LYRICS_STYLE)
                                lyrics_palette = gr.Dropdown(choices=CHOICES_LYRICS_PALETTE, value="tiktok_yellow", label=LABEL_LYRICS_PALETTE)
                                lyrics_position = gr.Dropdown(choices=CHOICES_LYRICS_POSITION, value="bottom", label=LABEL_LYRICS_POSITION)

                        process_btn = gr.Button('🎬 Create Music Video', variant='primary', size='lg')

                    with gr.Column(scale=1):
                        gr.Markdown('### 📺 Output')
                        status_output = gr.Textbox(label='Status', interactive=False, value=get_ready_status(python_status, cuda_status, MAX_THREADS, CPU_COUNT, ffmpeg_status, GPU_AVAILABLE, gpu_info, NVENC_AVAILABLE), lines=4, max_lines=4, elem_id='status-output-box')
                        video_output = gr.Video(label='Generated Music Video', interactive=False, elem_id='generated-video-output')
                        
                        create_shorts_from_mv_btn = gr.Button(
                            BTN_GENERATE_FROM_GENERATED,
                            variant='secondary',
                            size='lg',
                            elem_id='create-shorts-from-mv-btn'
                        )

            # ============================================================================
            # TAB 2: TIKTOK & SHORTS AUTO-CLIPPER (9:16)
            # ============================================================================
            with gr.Tab(TAB_TIKTOK_SHORTS, id="tiktok_shorts_tab"):
                gr.Markdown(f"## {TIKTOK_SECTION_TITLE}")
                gr.Markdown(TIKTOK_SECTION_DESC)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown('### 📁 Input Video File')
                        shorts_video_input = gr.File(
                            label=LABEL_SHORTS_INPUT_VIDEO,
                            file_types=['.mp4', '.mkv', '.mov'],
                            type='filepath',
                            elem_id='shorts-video-input'
                        )

                        with gr.Group():
                            gr.Markdown('### ⚙️ Clip Configuration & Duration')
                            shorts_clip_count = gr.Slider(
                                minimum=1, maximum=5, value=3, step=1,
                                label=LABEL_CLIP_COUNT, info=INFO_CLIP_COUNT
                            )
                            shorts_duration_mode = gr.Dropdown(
                                choices=CHOICES_DURATION_MODE,
                                value="auto_15_30",
                                label=LABEL_DURATION_MODE,
                            )

                        with gr.Group():
                            gr.Markdown('### 📐 9:16 Framing & AI Strategy')
                            shorts_framing_mode = gr.Radio(
                                choices=CHOICES_FRAMING_MODE,
                                value="smart_crop",
                                label=LABEL_FRAMING_MODE,
                            )
                            shorts_ai_strategy = gr.Radio(
                                choices=CHOICES_AI_STRATEGY,
                                value="smart_viral",
                                label=LABEL_AI_STRATEGY,
                            )
                            with gr.Row():
                                shorts_enable_qwen = gr.Checkbox(
                                    value=True,
                                    label=LABEL_ENABLE_QWEN,
                                    info=INFO_ENABLE_QWEN,
                                )
                                shorts_ai_vision_tier = gr.Dropdown(
                                    choices=CHOICES_AI_VISION_TIER,
                                    value="standard_2b",
                                    label=LABEL_AI_VISION_TIER,
                                    info=INFO_AI_VISION_TIER,
                                )

                        with gr.Accordion(GROUP_KARAOKE_TITLE, open=False):
                            shorts_enable_subtitles = gr.Checkbox(label=LABEL_ENABLE_KARAOKE, value=False, info=INFO_ENABLE_KARAOKE)
                            with gr.Row():
                                shorts_lyrics_mode = gr.Radio(choices=CHOICES_LYRICS_MODE, value="auto_whisper", label=LABEL_LYRICS_MODE)
                                shorts_lyrics_file = gr.File(label=LABEL_LYRICS_FILE, file_types=['.lrc', '.srt', '.ass', '.txt'], type='filepath')
                            shorts_lyrics_text = gr.Textbox(label=LABEL_LYRICS_TEXT, placeholder="Paste full song lyrics here...", lines=3, info=INFO_LYRICS_TEXT)
                            with gr.Row():
                                shorts_lyrics_style = gr.Dropdown(choices=CHOICES_LYRICS_STYLE, value="tiktok_bounce", label=LABEL_LYRICS_STYLE)
                                shorts_lyrics_palette = gr.Dropdown(choices=CHOICES_LYRICS_PALETTE, value="tiktok_yellow", label=LABEL_LYRICS_PALETTE)
                                shorts_lyrics_position = gr.Dropdown(choices=CHOICES_LYRICS_POSITION, value="bottom", label=LABEL_LYRICS_POSITION)

                        with gr.Group():
                            gr.Markdown('### 🎬 Rendering Mode')
                            if NVENC_AVAILABLE:
                                shorts_proc_mode = gr.Radio(
                                    choices=[('NVIDIA NVENC H.264', 'h264_nvenc'), ('NVIDIA NVENC HEVC', 'hevc_nvenc'), ('CPU (H.264)', 'cpu')],
                                    value='h264_nvenc',
                                    label=LABEL_PROCESSING_MODE
                                )
                            else:
                                shorts_proc_mode = gr.Radio(
                                    choices=[('CPU (H.264)', 'cpu')],
                                    value='cpu',
                                    label=LABEL_PROCESSING_MODE
                                )

                        shorts_process_btn = gr.Button(BTN_GENERATE_SHORTS, variant='primary', size='lg')

                    with gr.Column(scale=1):
                        gr.Markdown('### 📺 Results & 9:16 Vertical Clips')
                        shorts_status_output = gr.Textbox(
                            label=LABEL_SHORTS_OUTPUT_STATUS,
                            interactive=False,
                            value="Ready. Load a video or use the button from the main generator and click 'Generate 9:16 Clips'.",
                            lines=3,
                            max_lines=3,
                            elem_id='shorts-status-box'
                        )
                        shorts_summary_md = gr.Markdown("")
                        
                        with gr.Tabs():
                            with gr.TabItem("Clip #1"):
                                clip_v1 = gr.Video(label="Clip #1 (9:16)", interactive=False, elem_classes=['shorts-video-player'])
                            with gr.TabItem("Clip #2"):
                                clip_v2 = gr.Video(label="Clip #2 (9:16)", interactive=False, elem_classes=['shorts-video-player'])
                            with gr.TabItem("Clip #3"):
                                clip_v3 = gr.Video(label="Clip #3 (9:16)", interactive=False, elem_classes=['shorts-video-player'])
                            with gr.TabItem("Clip #4"):
                                clip_v4 = gr.Video(label="Clip #4 (9:16)", interactive=False, elem_classes=['shorts-video-player'])
                            with gr.TabItem("Clip #5"):
                                clip_v5 = gr.Video(label="Clip #5 (9:16)", interactive=False, elem_classes=['shorts-video-player'])

        # ============================================================================
        # EVENT BINDINGS
        # ============================================================================
        process_btn.click(
            fn=process_video,
            inputs=[
                audio_input, video_input,
                output_filename, processing_mode, custom_fps,
                mv_ai_vision_tier,
                enable_subtitles, lyrics_mode, lyrics_text, lyrics_file,
                lyrics_style, lyrics_palette, lyrics_position,
                session_state
            ],
            outputs=[video_output, status_output, session_state],
            show_progress='hidden'
        )

        create_shorts_from_mv_btn.click(
            fn=lambda vid: (vid, gr.Tabs(selected="tiktok_shorts_tab")),
            inputs=[video_output],
            outputs=[shorts_video_input, main_tabs],
        )

        shorts_process_btn.click(
            fn=process_social_clips,
            inputs=[
                shorts_video_input,
                shorts_clip_count,
                shorts_duration_mode,
                shorts_framing_mode,
                shorts_ai_strategy,
                shorts_enable_qwen,
                shorts_ai_vision_tier,
                shorts_proc_mode,
                shorts_enable_subtitles,
                shorts_lyrics_mode,
                shorts_lyrics_text,
                shorts_lyrics_file,
                shorts_lyrics_style,
                shorts_lyrics_palette,
                shorts_lyrics_position,
            ],
            outputs=[
                shorts_status_output,
                clip_v1, clip_v2, clip_v3, clip_v4, clip_v5,
                shorts_summary_md,
            ],
            show_progress='hidden'
        )


    return app


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Clean up old files only on startup
    cleanup_on_startup()
    
    app = create_ui()
    launch_port = find_launch_port()
    app.launch(
        server_name="127.0.0.1",
        server_port=launch_port,
        share=False,
        inbrowser=True,
        show_error=True
    )

