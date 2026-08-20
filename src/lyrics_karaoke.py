#!/usr/bin/env python3
"""
Lyrics & Karaoke Module for BeatSync Engine
- Automatic voice transcription & word-level timing with Whisper
- User lyrics text alignment with audio vocals & beats
- Advanced SubStation Alpha (.ass) generation with TikTok-style animated word-bounce & karaoke sweep
- Burning subtitles directly into 16:9 and vertical 9:16 videos via FFmpeg
- Exporting synced .ass, .srt, and .lrc subtitle files
"""

import os
import sys
import re
import math
import time
import shutil
import tempfile
import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable

# Project imports
from logger import setup_environment, ROOT_DIR
setup_environment()

from paths import get_processing_dir, get_subtitles_output_dir
from ffmpeg_processing import (
    FFMPEG_PATH,
    get_video_duration,
    get_video_fps,
    get_video_resolution,
    has_audio_stream,
    extract_audio_from_video,
    get_nvenc_quality_args,
    get_cpu_h264_quality_args,
    _run_media_command,
    _short_ffmpeg_error,
)
from gpu_cpu_utils import GPU_AVAILABLE, NVENC_AVAILABLE


@dataclass
class TimedWord:
    word: str
    start: float
    end: float
    confidence: float = 1.0


@dataclass
class TimedPhrase:
    words: List[TimedWord]
    start: float
    end: float
    text: str = ""

    def __post_init__(self):
        if not self.text and self.words:
            self.text = " ".join(w.word for w in self.words)


# Color Palettes in ASS BGR Hex format (&HAABBGGRR&)
# In ASS format, colors are &H[Alpha][Blue][Green][Red]&
COLOR_PALETTES = {
    "tiktok_yellow": {
        "name": "TikTok Yellow (Giallo Vibrante)",
        "active_primary": "&H0000D7FF&",    # RGB #FFD700 -> BGR 00 D7 FF
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00000000&",           # Black
        "shadow": "&H80000000&",            # Semi-transparent Black
    },
    "neon_cyan": {
        "name": "Neon Cyan (Azzurro Elettrico)",
        "active_primary": "&H00FFFF00&",    # RGB #00FFFF -> BGR FF FF 00
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00201005&",           # Deep Navy Outline
        "shadow": "&H80000000&",
    },
    "hot_pink": {
        "name": "Hot Pink (Fucsia / Magenta)",
        "active_primary": "&H00852AFF&",    # RGB #FF2A85 -> BGR 85 2A FF
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00200020&",           # Dark Magenta Outline
        "shadow": "&H80000000&",
    },
    "cyber_green": {
        "name": "Cyber Green (Verde Lime)",
        "active_primary": "&H0014FF39&",    # RGB #39FF14 -> BGR 14 FF 39
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00002000&",           # Dark Green Outline
        "shadow": "&H80000000&",
    },
    "flame_orange": {
        "name": "Flame Orange (Arancione Fuoco)",
        "active_primary": "&H000066FF&",    # RGB #FF6600 -> BGR 00 66 FF
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00000A1A&",           # Dark Outline
        "shadow": "&H80000000&",
    },
    "pure_white": {
        "name": "Pure White / Monochrome",
        "active_primary": "&H00FFFFFF&",    # White
        "base_primary": "&H00A0A0A0&",      # Muted Gray
        "outline": "&H00000000&",           # Black
        "shadow": "&H80000000&",
    },
}


def _clean_text_word(raw: str) -> str:
    return re.sub(r"[^\w\s\'-]", "", raw).strip()


def _format_ass_timestamp(seconds: float) -> str:
    """Convert floating seconds to ASS timestamp format H:MM:SS.cc"""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centis = int(round((secs - int(secs)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centis:02d}"


def _format_lrc_timestamp(seconds: float) -> str:
    """Convert floating seconds to LRC timestamp format [mm:ss.xx]"""
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    secs = seconds % 60
    centis = int(round((secs - int(secs)) * 100))
    if centis >= 100:
        centis = 99
    return f"[{minutes:02d}:{int(secs):02d}.{centis:02d}]"


def _format_srt_timestamp(seconds: float) -> str:
    """Convert floating seconds to SRT timestamp format HH:MM:SS,mmm"""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ============================================================================
# TRANSCRIPTION & ALIGNMENT
# ============================================================================

def transcribe_audio_whisper(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[TimedWord]:
    """
    Transcribe audio with faster-whisper and extract word-level timestamps.
    """
    if not os.path.exists(audio_path):
        return []

    try:
        from faster_whisper import WhisperModel

        if progress_callback:
            progress_callback(f"Caricamento modello Whisper AI ({model_size})...")

        # Use GPU CUDA if available, fallback to CPU
        device = "cuda" if GPU_AVAILABLE else "cpu"
        compute_type = "float16" if GPU_AVAILABLE else "int8"

        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception:
            # Fallback to CPU float32 if specific CUDA compute type fails
            model = WhisperModel(model_size, device="cpu", compute_type="int8")

        if progress_callback:
            progress_callback("Trascrizione vocale e calcolo timestamp parole in corso...")

        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            language=language if language and language != "auto" else None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )

        timed_words: List[TimedWord] = []
        for segment in segments:
            if hasattr(segment, "words") and segment.words:
                for w in segment.words:
                    word_str = w.word.strip()
                    if word_str:
                        timed_words.append(TimedWord(
                            word=word_str,
                            start=float(w.start),
                            end=float(w.end),
                            confidence=float(getattr(w, "probability", 1.0)),
                        ))

        return timed_words
    except Exception as e:
        print(f"   ⚠️  Whisper transcription error: {e}")
        return []


def align_user_lyrics_with_audio(
    provided_lyrics_text: str,
    audio_words: List[TimedWord],
    audio_duration: float = 0.0,
    beat_times: Optional[List[float]] = None,
) -> List[TimedWord]:
    """
    Align user-provided lyrics text with audio timestamps:
    - If Whisper audio_words are available, uses fuzzy matching to map user words
      onto Whisper's exact timing boundaries.
    - If Whisper words are empty, distributes lines across beat grid / duration.
    """
    raw_lines = [line.strip() for line in provided_lyrics_text.splitlines() if line.strip()]
    if not raw_lines:
        return audio_words

    user_words: List[str] = []
    for line in raw_lines:
        words_in_line = line.split()
        user_words.extend(words_in_line)

    if not user_words:
        return audio_words

    # If we have Whisper word timestamps, align user words with audio words
    if audio_words and len(audio_words) > 0:
        whisper_clean = [_clean_text_word(w.word).lower() for w in audio_words]
        user_clean = [_clean_text_word(w).lower() for w in user_words]

        matcher = difflib.SequenceMatcher(None, whisper_clean, user_clean)
        aligned_words: List[TimedWord] = []

        last_end = 0.0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for wi, uj in zip(range(i1, i2), range(j1, j2)):
                    tw = audio_words[wi]
                    aligned_words.append(TimedWord(
                        word=user_words[uj],  # Exact user spelling and case
                        start=tw.start,
                        end=tw.end,
                        confidence=1.0,
                    ))
                    last_end = tw.end

            elif tag in ('replace', 'insert', 'delete'):
                # Handle replaced or inserted words by interpolating time window
                t_start = audio_words[i1].start if i1 < len(audio_words) else last_end
                t_end = audio_words[min(i2, len(audio_words) - 1)].end if i2 <= len(audio_words) and i1 < len(audio_words) else (t_start + 2.0)
                span_duration = max(0.2, t_end - t_start)

                u_count = j2 - j1
                if u_count > 0:
                    dt = span_duration / float(u_count)
                    for k, uj in enumerate(range(j1, j2)):
                        w_s = t_start + k * dt
                        w_e = min(t_end, w_s + dt)
                        aligned_words.append(TimedWord(
                            word=user_words[uj],
                            start=w_s,
                            end=w_e,
                            confidence=0.85,
                        ))
                        last_end = w_e

        if aligned_words:
            return aligned_words

    # Fallback: Distribute provided lyrics over beat times or song duration
    if audio_duration <= 0.0:
        audio_duration = 30.0

    aligned_words = []
    total_words = len(user_words)
    time_per_word = max(0.25, (audio_duration * 0.85) / max(1, total_words))
    start_offset = min(2.0, audio_duration * 0.05)

    for i, w in enumerate(user_words):
        w_start = start_offset + i * time_per_word
        w_end = min(audio_duration, w_start + time_per_word * 0.9)
        aligned_words.append(TimedWord(
            word=w,
            start=w_start,
            end=w_end,
            confidence=0.7,
        ))

    return aligned_words


def parse_lrc_or_srt(text_or_path: str) -> List[TimedPhrase]:
    """Parse a .lrc or .srt file or text into TimedPhrase objects."""
    content = text_or_path
    if os.path.isfile(text_or_path):
        try:
            with open(text_or_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return []

    phrases: List[TimedPhrase] = []

    # Check for LRC format [mm:ss.xx]
    lrc_matches = re.findall(r"\[(\d{2}):(\d{2}(?:\.\d+)?)\](.*)", content)
    if lrc_matches:
        timed_lines = []
        for m, s, txt in lrc_matches:
            t = int(m) * 60 + float(s)
            text_line = txt.strip()
            if text_line:
                timed_lines.append((t, text_line))

        timed_lines.sort(key=lambda x: x[0])
        for idx, (t_start, txt) in enumerate(timed_lines):
            if idx + 1 < len(timed_lines):
                t_end = min(timed_lines[idx + 1][0], t_start + 6.0)
            else:
                t_end = t_start + 4.0

            words_str = txt.split()
            if not words_str:
                continue
            dt = max(0.1, (t_end - t_start) / len(words_str))
            words = [
                TimedWord(w, t_start + k * dt, min(t_end, t_start + (k + 1) * dt))
                for k, w in enumerate(words_str)
            ]
            phrases.append(TimedPhrase(words=words, start=t_start, end=t_end, text=txt))
        return phrases

    # Check for SRT format
    srt_blocks = re.split(r"\n\s*\n", content.strip())
    for block in srt_blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) >= 2:
            time_line = lines[1] if re.match(r"^\d+$", lines[0]) and len(lines) >= 3 else lines[0]
            txt_lines = lines[2:] if re.match(r"^\d+$", lines[0]) and len(lines) >= 3 else lines[1:]
            time_match = re.search(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})", time_line)
            if time_match:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = [int(v) for v in time_match.groups()]
                t_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                t_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                full_text = " ".join(txt_lines)
                words_str = full_text.split()
                if words_str:
                    dt = (t_end - t_start) / len(words_str)
                    words = [
                        TimedWord(w, t_start + k * dt, min(t_end, t_start + (k + 1) * dt))
                        for k, w in enumerate(words_str)
                    ]
                    phrases.append(TimedPhrase(words=words, start=t_start, end=t_end, text=full_text))

    return phrases


def group_words_into_phrases(
    words: List[TimedWord],
    max_words_per_line: int = 4,
    max_gap_seconds: float = 1.0,
    force_all_caps: bool = False,
) -> List[TimedPhrase]:
    """
    Group timed words into dynamic short phrases (2 to 5 words)
    ideal for TikTok high-impact readability.
    """
    if not words:
        return []

    phrases: List[TimedPhrase] = []
    current_words: List[TimedWord] = []

    for idx, w in enumerate(words):
        w_text = w.word.upper() if force_all_caps else w.word
        clean_word = TimedWord(word=w_text, start=w.start, end=w.end, confidence=w.confidence)

        if not current_words:
            current_words.append(clean_word)
            continue

        gap = clean_word.start - current_words[-1].end
        is_punctuation = current_words[-1].word.endswith((".", "!", "?", ",", ";"))

        if len(current_words) >= max_words_per_line or gap > max_gap_seconds or (is_punctuation and len(current_words) >= 2):
            p_start = current_words[0].start
            p_end = max(current_words[-1].end, current_words[0].start + 0.5)
            phrases.append(TimedPhrase(words=list(current_words), start=p_start, end=p_end))
            current_words = [clean_word]
        else:
            current_words.append(clean_word)

    if current_words:
        p_start = current_words[0].start
        p_end = max(current_words[-1].end, current_words[0].start + 0.5)
        phrases.append(TimedPhrase(words=list(current_words), start=p_start, end=p_end))

    return phrases


# ============================================================================
# ADVANCED SUBSTATION ALPHA (.ASS) GENERATION
# ============================================================================

def generate_karaoke_ass(
    phrases: List[TimedPhrase],
    output_ass_path: str,
    video_width: int = 1920,
    video_height: int = 1080,
    animation_style: str = "tiktok_bounce",
    palette_key: str = "tiktok_yellow",
    font_family: str = "Arial Black",
    font_size: int = None,
    position_mode: str = "bottom",
    uppercase: bool = True,
) -> str:
    r"""
    Generate a professional Advanced SubStation Alpha (.ass) subtitle file.
    Styles:
    - 'tiktok_bounce': Active word glows in vibrant color with bounce / scale emphasis.
    - 'karaoke_sweep': Syllables / words fill with color smoothly via \kf tags.
    - 'clean_pop': Clean modern subtitles without distracting transitions.
    """

    palette = COLOR_PALETTES.get(palette_key, COLOR_PALETTES["tiktok_yellow"])
    is_vertical = video_height > video_width

    # Optimal font sizing based on resolution & aspect ratio
    if font_size is None or font_size <= 0:
        if is_vertical:
            font_size = int(round(video_height * 0.038))  # ~72px on 1080x1920
        else:
            font_size = int(round(video_height * 0.055))  # ~60px on 1920x1080

    # Vertical positioning (MarginV in ASS)
    if position_mode == "center":
        alignment = 5  # Center Middle
        margin_v = 0
    elif position_mode == "top":
        alignment = 8  # Top Center
        margin_v = int(round(video_height * 0.15))
    else:
        # Default 'bottom'
        alignment = 2  # Bottom Center
        margin_v = int(round(video_height * 0.18)) if is_vertical else int(round(video_height * 0.12))

    outline_size = max(3, int(round(font_size * 0.08)))
    shadow_size = max(2, int(round(font_size * 0.04)))

    ass_header = f"""[Script Info]
; Script generated by BeatSync Engine Lyrics & Karaoke Module
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{font_size},{palette['base_primary']},{palette['active_primary']},{palette['outline']},{palette['shadow']},-1,0,0,0,100,100,1,0,1,{outline_size},{shadow_size},{alignment},40,40,{margin_v},1
Style: Highlight,{font_family},{font_size},{palette['active_primary']},{palette['base_primary']},{palette['outline']},{palette['shadow']},-1,0,0,0,100,100,1,0,1,{outline_size + 1},{shadow_size},{alignment},40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: List[str] = []

    for phrase in phrases:
        if not phrase.words:
            continue

        p_start_ass = _format_ass_timestamp(phrase.start)
        p_end_ass = _format_ass_timestamp(phrase.end)

        # Style A: TikTok Bounce / Active Word Highlight
        if animation_style == "tiktok_bounce":
            # For each word in the phrase, create a micro-event where that specific word is highlighted
            for i, target_word in enumerate(phrase.words):
                w_start_ass = _format_ass_timestamp(target_word.start)
                w_end_ass = _format_ass_timestamp(target_word.end)

                text_parts = []
                for j, w in enumerate(phrase.words):
                    w_text = w.word.upper() if uppercase else w.word
                    if j == i:
                        # Highlighted active word: vibrant color + slight scale animation
                        text_parts.append(
                            f"{{\\c{palette['active_primary']}\\fscx112\\fscy112\\t(0,80,\\fscx100\\fscy100)}}{w_text}{{\\r}}"
                        )
                    else:
                        # Inactive words: crisp base white color
                        text_parts.append(f"{{\\c{palette['base_primary']}}}{w_text}{{\\r}}")

                full_line = " ".join(text_parts)
                events.append(f"Dialogue: 0,{w_start_ass},{w_end_ass},Default,,0,0,0,,{full_line}")

        # Style B: Classic Karaoke Sweep with \kf
        elif animation_style == "karaoke_sweep":
            karaoke_parts = []
            for w in phrase.words:
                w_text = w.word.upper() if uppercase else w.word
                # Duration in centiseconds
                dur_cs = max(1, int(round((w.end - w.start) * 100)))
                karaoke_parts.append(f"{{\\kf{dur_cs}}}{w_text}")

            full_line = " ".join(karaoke_parts)
            events.append(f"Dialogue: 0,{p_start_ass},{p_end_ass},Default,,0,0,0,,{full_line}")

        # Style C: Clean Pop-in Text
        else:
            text_clean = phrase.text.upper() if uppercase else phrase.text
            events.append(f"Dialogue: 0,{p_start_ass},{p_end_ass},Default,,0,0,0,,{text_clean}")

    ass_content = ass_header + "\n".join(events) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    return output_ass_path


def export_subtitles_bundle(
    phrases: List[TimedPhrase],
    base_name: str,
    output_dir: str = None,
    uppercase: bool = False,
) -> Dict[str, str]:
    """Export .ass, .srt, and .lrc files into output directory."""
    if output_dir is None:
        output_dir = get_subtitles_output_dir()
    os.makedirs(output_dir, exist_ok=True)

    # 1. SRT format
    srt_path = os.path.join(output_dir, f"{base_name}.srt")
    srt_lines = []
    for idx, p in enumerate(phrases, 1):
        txt = p.text.upper() if uppercase else p.text
        srt_lines.append(f"{idx}\n{_format_srt_timestamp(p.start)} --> {_format_srt_timestamp(p.end)}\n{txt}\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    # 2. LRC format
    lrc_path = os.path.join(output_dir, f"{base_name}.lrc")
    lrc_lines = [f"[ti:{base_name}]", "[by:BeatSync Engine]"]
    for p in phrases:
        txt = p.text.upper() if uppercase else p.text
        lrc_lines.append(f"{_format_lrc_timestamp(p.start)}{txt}")
    with open(lrc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lrc_lines))

    # 3. Default ASS format
    ass_path = os.path.join(output_dir, f"{base_name}.ass")
    generate_karaoke_ass(phrases, ass_path, uppercase=uppercase)

    return {
        "srt": srt_path,
        "lrc": lrc_path,
        "ass": ass_path,
    }


# ============================================================================
# VIDEO SUBTITLE BURNING
# ============================================================================

def burn_karaoke_to_video(
    input_video: str,
    ass_path: str,
    output_video: str,
    use_nvenc: bool = False,
    gpu_encoder: str = "h264_nvenc",
    custom_fps: float = None,
) -> Tuple[bool, str]:
    """
    Burn ASS karaoke subtitles into video using FFmpeg with hardware acceleration.
    """
    if not os.path.exists(input_video):
        return False, f"Input video not found: {input_video}"
    if not os.path.exists(ass_path):
        return False, f"Subtitle file not found: {ass_path}"

    try:
        fps = custom_fps if custom_fps and custom_fps > 0 else get_video_fps(input_video)
        effective_use_nvenc = use_gpu_mode = use_nvenc and NVENC_AVAILABLE and gpu_encoder != "cpu" and gpu_encoder != "none"

        # Escape path for FFmpeg subtitles filter on Windows
        # e.g., 'C\:/path/to/sub.ass' with colons and backslashes escaped
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")

        vf_filter = f"ass='{escaped_ass}'"
        audio_available = has_audio_stream(input_video)

        base_cmd = [
            FFMPEG_PATH,
            '-nostdin',
            '-hide_banner',
            '-i', input_video,
            '-vf', vf_filter,
        ]

        if audio_available:
            base_cmd.extend(['-c:a', 'copy'])
        else:
            base_cmd.extend(['-an'])

        tail_cmd = [
            '-fps_mode', 'cfr',
            '-r', str(fps),
            '-fflags', '+genpts',
            '-movflags', '+faststart',
            '-y',
            output_video,
        ]

        # 1. Try NVENC if available
        if effective_use_nvenc:
            cmd_nvenc = base_cmd + get_nvenc_quality_args(gpu_encoder, include_pix_fmt=True) + tail_cmd
            res = _run_media_command(cmd_nvenc, timeout=300)
            if res.returncode == 0 and os.path.exists(output_video) and os.path.getsize(output_video) > 0:
                return True, ""

        # 2. CPU fallback
        cmd_cpu = base_cmd + get_cpu_h264_quality_args(include_pix_fmt=True) + tail_cmd
        res = _run_media_command(cmd_cpu, timeout=300)
        if res.returncode == 0 and os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            return True, ""

        err = _short_ffmpeg_error(res.stderr, 800)
        return False, f"FFmpeg burning error (code {res.returncode}): {err}"

    except Exception as e:
        return False, str(e)
