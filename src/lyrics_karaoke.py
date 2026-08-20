#!/usr/bin/env python3
"""
Lyrics & Karaoke Synchronization Module for BeatSync Engine
- Frame-accurate AI speech/lyrics transcription (faster-whisper)
- Line-by-line vocal forced alignment
- TikTok-style animated word-by-word bouncing captions & karaoke sweeps
- Advanced SubStation Alpha (.ass) generation & FFmpeg burning
"""

import os
import sys
import re
import difflib
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable, Any

# Project imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from logger import (
    setup_environment,
    FFMPEG_EXE as FFMPEG_PATH,
)
from gpu_cpu_utils import GPU_AVAILABLE, NVENC_AVAILABLE
from paths import get_subtitles_output_dir

setup_environment()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

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


# Color Palettes in ASS BGR Hex format (&H00BBGGRR&)
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
        "outline": "&H00052005&",
        "shadow": "&H80000000&",
    },
    "flame_orange": {
        "name": "Flame Orange (Arancione Fuoco)",
        "active_primary": "&H000066FF&",    # RGB #FF6600 -> BGR 00 66 FF
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00001030&",
        "shadow": "&H80000000&",
    },
    "pure_white": {
        "name": "Pure White (Bianco Pulito)",
        "active_primary": "&H0000D7FF&",    # Highlight in Gold
        "base_primary": "&H00FFFFFF&",      # White
        "outline": "&H00000000&",
        "shadow": "&H80000000&",
    },
}


# ============================================================================
# STRING & TIME HELPERS
# ============================================================================

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
    """Convert floating seconds to LRC timestamp format [MM:SS.xx]"""
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


def _clean_text_word(w: str) -> str:
    """Clean punctuation from a word for fuzzy matching."""
    return re.sub(r"[^\w\s]", "", w).strip()


# ============================================================================
# TRANSCRIPTION & ALIGNMENT
# ============================================================================

_WHISPER_MODEL_CACHE: Dict[str, Any] = {}


def get_cached_whisper_model(model_size: str = "tiny") -> Any:
    """Load and cache WhisperModel in memory for instant reuse."""
    global _WHISPER_MODEL_CACHE
    if model_size not in _WHISPER_MODEL_CACHE:
        from faster_whisper import WhisperModel
        try:
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception:
            model = WhisperModel(model_size, device="cpu", compute_type="float32")
        _WHISPER_MODEL_CACHE[model_size] = model
    return _WHISPER_MODEL_CACHE[model_size]


def extract_vocal_filtered_audio(input_media: str, output_wav: str) -> bool:
    """
    Extract audio with vocal-range bandpass and normalization:
    - High-pass at 200Hz (removes heavy bass and sub-bass kicks)
    - Low-pass at 4500Hz (removes harsh cymbals/high noise)
    - Normalization for clear vocal formants
    """
    try:
        cmd = [
            FFMPEG_PATH,
            '-nostdin',
            '-hide_banner',
            '-i', input_media,
            '-af', 'highpass=f=200,lowpass=f=4500,dynaudnorm=f=150:g=15',
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            '-y',
            output_wav,
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=60)
        return res.returncode == 0 and os.path.exists(output_wav) and os.path.getsize(output_wav) > 0
    except Exception:
        return False


def transcribe_audio_whisper(
    audio_path: str,
    model_size: str = "tiny",
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[TimedWord]:
    """
    Transcribe audio with faster-whisper and extract word-level timestamps.
    Uses vocal-enhanced audio and disables VAD filtering for singing music.
    """
    if not os.path.exists(audio_path):
        return []

    temp_vocal_wav = None
    try:
        if progress_callback:
            progress_callback(f"Caricamento modello Whisper AI ({model_size})...")

        model = get_cached_whisper_model(model_size)

        # Prepare vocal-filtered audio
        temp_vocal_wav = os.path.join(tempfile.gettempdir(), f"whisper_vocal_{os.path.basename(audio_path)}.wav")
        if extract_vocal_filtered_audio(audio_path, temp_vocal_wav):
            target_audio = temp_vocal_wav
        else:
            target_audio = audio_path

        if progress_callback:
            progress_callback("Trascrizione vocale AI e rilevamento parole...")

        prompt_str = initial_prompt[:400] if initial_prompt else None

        segments, info = model.transcribe(
            target_audio,
            word_timestamps=True,
            language=language if language and language != "auto" else None,
            initial_prompt=prompt_str,
            vad_filter=False,  # Essential for singing over background music!
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,  # Prevents repeating hallucinations on musical loops
        )

        total_dur = getattr(info, "duration", 0.0) or 0.0
        timed_words: List[TimedWord] = []

        for segment in segments:
            if progress_callback and total_dur > 0:
                progress_callback(f"Trascrizione Whisper: {int(segment.start)}s / {int(total_dur)}s ({len(timed_words)} parole)...")

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

        if progress_callback:
            progress_callback(f"Trascrizione completata: {len(timed_words)} parole rilevate.")

        return timed_words
    except Exception as e:
        print(f"   ⚠️  Whisper transcription error: {e}")
        return []
    finally:
        if temp_vocal_wav and os.path.exists(temp_vocal_wav):
            try:
                os.remove(temp_vocal_wav)
            except Exception:
                pass


def align_user_lyrics_with_audio(
    provided_lyrics_text: str,
    audio_words: Optional[List[TimedWord]] = None,
    audio_duration: float = 0.0,
    beat_times: Optional[List[float]] = None,
) -> List[TimedWord]:
    """
    Line-by-line robust alignment of user lyrics with audio timestamps:
    - Splits user lyrics into individual lines (verses/chorus).
    - If Whisper audio_words are available, maps each user line to the matching
      Whisper vocal cluster.
    - If audio_words is empty, aligns each line to musical beat intervals.
    """
    raw_lines = [line.strip() for line in provided_lyrics_text.splitlines() if line.strip()]
    if not raw_lines:
        return audio_words or []

    # Case 1: Whisper audio words are available
    if audio_words and len(audio_words) > 0:
        whisper_clean = [_clean_text_word(w.word).lower() for w in audio_words]
        all_user_words = []
        for line in raw_lines:
            all_user_words.extend(line.split())

        user_clean = [_clean_text_word(w).lower() for w in all_user_words]

        matcher = difflib.SequenceMatcher(None, whisper_clean, user_clean)
        aligned_words: List[TimedWord] = []

        last_end = 0.0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for wi, uj in zip(range(i1, i2), range(j1, j2)):
                    tw = audio_words[wi]
                    aligned_words.append(TimedWord(
                        word=all_user_words[uj],  # Exact user spelling and case
                        start=tw.start,
                        end=tw.end,
                        confidence=1.0,
                    ))
                    last_end = tw.end

            elif tag in ('replace', 'insert', 'delete'):
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
                            word=all_user_words[uj],
                            start=w_s,
                            end=w_e,
                            confidence=0.85,
                        ))
                        last_end = w_e

        if aligned_words:
            return aligned_words

    # Case 2: Beat-Grid Alignment (Line-by-line over musical beats)
    all_user_words = []
    for line in raw_lines:
        all_user_words.extend(line.split())

    if beat_times is not None and len(beat_times) >= 4:
        beats_clean = [float(b) for b in beat_times if float(b) >= 0.0]
        if len(beats_clean) >= 4:
            total_beats = len(beats_clean)
            total_words = len(all_user_words)
            aligned_words = []

            step = max(1.0, float(total_beats - 2) / max(1, total_words))
            for i, w in enumerate(all_user_words):
                beat_idx = int(round(1 + i * step))
                beat_idx = min(beat_idx, total_beats - 2)
                w_start = beats_clean[beat_idx]
                next_b = beats_clean[min(total_beats - 1, beat_idx + 1)]
                w_end = min(w_start + max(0.3, (next_b - w_start) * 0.9), w_start + 2.5)
                aligned_words.append(TimedWord(
                    word=w,
                    start=w_start,
                    end=w_end,
                    confidence=0.9,
                ))
            return aligned_words

    # Case 3: Fallback linear distribution over duration
    if audio_duration <= 0.0:
        audio_duration = 30.0

    aligned_words = []
    total_words = len(all_user_words)
    start_offset = min(1.5, audio_duration * 0.05)
    usable_dur = max(2.0, audio_duration * 0.9 - start_offset)
    time_per_word = usable_dur / max(1, total_words)

    for i, w in enumerate(all_user_words):
        w_start = start_offset + i * time_per_word
        w_end = min(audio_duration, w_start + time_per_word * 0.9)
        aligned_words.append(TimedWord(
            word=w,
            start=w_start,
            end=w_end,
            confidence=0.75,
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
            time_match = re.search(
                r"(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,\.](\d{3})",
                lines[0] if "-->" in lines[0] else lines[1]
            )
            if time_match:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
                t_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                t_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                txt = " ".join(lines[1:] if "-->" in lines[0] else lines[2:]).strip()
                if txt:
                    words_str = txt.split()
                    dt = max(0.1, (t_end - t_start) / len(words_str))
                    words = [
                        TimedWord(w, t_start + k * dt, min(t_end, t_start + (k + 1) * dt))
                        for k, w in enumerate(words_str)
                    ]
                    phrases.append(TimedPhrase(words=words, start=t_start, end=t_end, text=txt))
    return phrases


def group_words_into_phrases(
    words: List[TimedWord],
    max_words_per_line: int = 4,
    max_gap_seconds: float = 0.45,
    force_all_caps: bool = False,
) -> List[TimedPhrase]:
    """
    Group timed words into dynamic natural lyric lines/phrases:
    - Splits on vocal pauses (> 0.45s) so words never bridge across singing silences.
    - Splits on punctuation (. , ! ? ;).
    - Keeps 2 to 5 words per line for optimal social media readability.
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
        prev_word = current_words[-1].word
        is_punctuation = prev_word.endswith((".", "!", "?", ",", ";", ":", "-"))

        # Split condition: natural pause > 0.45s, punctuation, or reached max words
        if gap > max_gap_seconds or len(current_words) >= max_words_per_line or (is_punctuation and len(current_words) >= 2):
            p_start = current_words[0].start
            p_end = max(current_words[-1].end, current_words[0].start + 0.3)
            phrases.append(TimedPhrase(words=list(current_words), start=p_start, end=p_end))
            current_words = [clean_word]
        else:
            current_words.append(clean_word)

    if current_words:
        p_start = current_words[0].start
        p_end = max(current_words[-1].end, current_words[0].start + 0.3)
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
    - 'tiktok_bounce': Continuous flicker-free phrase display with active word highlight & pop.
    - 'karaoke_sweep': Syllables / words fill with color smoothly via \kf tags.
    - 'clean_pop': Clean modern subtitles without distracting transitions.
    """
    palette = COLOR_PALETTES.get(palette_key, COLOR_PALETTES["tiktok_yellow"])
    is_vertical = (video_height > video_width)

    # Dynamic font sizing tailored for resolution
    if font_size is None or font_size <= 0:
        if is_vertical:
            font_size = max(38, int(round(video_height * 0.038)))  # ~72px for 1080x1920
        else:
            font_size = max(32, int(round(video_height * 0.052)))  # ~56px for 1920x1080

    # Alignment & Margin
    if position_mode == "top":
        alignment = 8  # Top Center
        margin_v = int(round(video_height * 0.12))
    elif position_mode == "center":
        alignment = 5  # Middle Center
        margin_v = int(round(video_height * 0.05))
    else:
        alignment = 2  # Bottom Center
        margin_v = int(round(video_height * 0.18)) if is_vertical else int(round(video_height * 0.12))

    outline_size = max(3, int(round(font_size * 0.09)))
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

        # Style A: TikTok Bounce / Active Word Highlight (Continuous & Flicker-Free)
        if animation_style == "tiktok_bounce":
            words_count = len(phrase.words)

            # 1. Lead-in event if phrase.start is before first word starts
            first_w_start = phrase.words[0].start
            if phrase.start < first_w_start - 0.05:
                text_all_base = " ".join(
                    f"{{\\1c{palette['base_primary']}}}{(w.word.upper() if uppercase else w.word)}{{\\r}}"
                    for w in phrase.words
                )
                events.append(f"Dialogue: 0,{_format_ass_timestamp(phrase.start)},{_format_ass_timestamp(first_w_start)},Default,,0,0,0,,{text_all_base}")

            # 2. Word-by-word micro-events (active word seamlessly transitions without disappearing)
            for i, target_word in enumerate(phrase.words):
                w_start = target_word.start
                # End of this word's highlight is when the next word starts (or phrase.end)
                if i + 1 < words_count:
                    w_end = phrase.words[i + 1].start
                else:
                    w_end = max(target_word.end, phrase.end)

                if w_end <= w_start:
                    w_end = w_start + 0.2

                w_start_ass = _format_ass_timestamp(w_start)
                w_end_ass = _format_ass_timestamp(w_end)

                text_parts = []
                for j, w in enumerate(phrase.words):
                    w_text = w.word.upper() if uppercase else w.word
                    if j == i:
                        # Highlighted active word: vibrant color + slight bounce scale
                        text_parts.append(
                            f"{{\\1c{palette['active_primary']}\\fscx112\\fscy112\\t(0,80,\\fscx100\\fscy100)}}{w_text}{{\\r}}"
                        )
                    else:
                        # Inactive words in the phrase: crisp white
                        text_parts.append(f"{{\\1c{palette['base_primary']}}}{w_text}{{\\r}}")

                full_line = " ".join(text_parts)
                events.append(f"Dialogue: 0,{w_start_ass},{w_end_ass},Default,,0,0,0,,{full_line}")

        # Style B: Classic Karaoke Sweep with \kf
        elif animation_style == "karaoke_sweep":
            karaoke_parts = []
            for w in phrase.words:
                w_text = w.word.upper() if uppercase else w.word
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


# ============================================================================
# EXPORT & BURN HELPERS
# ============================================================================

def export_subtitles_bundle(
    phrases: List[TimedPhrase],
    base_name: str,
    output_folder: str = None,
) -> Dict[str, str]:
    """Export synchronized lyrics in .ass, .srt, and .lrc formats."""
    if output_folder is None:
        output_folder = get_subtitles_output_dir()
    os.makedirs(output_folder, exist_ok=True)

    ass_path = os.path.join(output_folder, f"{base_name}.ass")
    srt_path = os.path.join(output_folder, f"{base_name}.srt")
    lrc_path = os.path.join(output_folder, f"{base_name}.lrc")

    # 1. ASS
    generate_karaoke_ass(phrases, ass_path)

    # 2. SRT
    srt_lines = []
    for idx, p in enumerate(phrases, start=1):
        s_start = _format_srt_timestamp(p.start)
        s_end = _format_srt_timestamp(p.end)
        srt_lines.append(f"{idx}\n{s_start} --> {s_end}\n{p.text}\n")
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_lines))

    # 3. LRC
    lrc_lines = []
    for p in phrases:
        lrc_lines.append(f"{_format_lrc_timestamp(p.start)}{p.text}")
    with open(lrc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lrc_lines))

    return {"ass": ass_path, "srt": srt_path, "lrc": lrc_path}


def burn_karaoke_to_video(
    input_video: str,
    ass_subtitle_file: str,
    output_video: str,
    use_nvenc: bool = False,
    gpu_encoder: str = "h264_nvenc",
) -> Tuple[bool, str]:
    """Burn .ass subtitle file directly into video using FFmpeg libass filter."""
    if not os.path.exists(input_video) or not os.path.exists(ass_subtitle_file):
        return False, "Input video or subtitle file missing"

    escaped_ass = ass_subtitle_file.replace("\\", "/").replace(":", "\\:")
    vf_arg = f"ass='{escaped_ass}'"

    cmd = [
        FFMPEG_PATH,
        '-nostdin',
        '-hide_banner',
        '-i', input_video,
        '-vf', vf_arg,
        '-c:a', 'copy',
        '-y',
        output_video,
    ]

    if use_nvenc and NVENC_AVAILABLE:
        cmd.extend(['-c:v', gpu_encoder, '-preset', 'p6', '-cq', '18'])
    else:
        cmd.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '18'])

    res = subprocess.run(cmd, capture_output=True, timeout=300)
    if res.returncode == 0 and os.path.exists(output_video) and os.path.getsize(output_video) > 0:
        return True, ""
    return False, f"FFmpeg burn failed: {res.stderr.decode('utf-8', errors='ignore')}"
