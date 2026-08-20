#!/usr/bin/env python3
"""
Test suite for Lyrics & Karaoke Module
"""

import os
import sys
import tempfile
import numpy as np
import cv2

# Project imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from logger import setup_environment
setup_environment()

from lyrics_karaoke import (
    TimedWord,
    TimedPhrase,
    group_words_into_phrases,
    align_user_lyrics_with_audio,
    parse_lrc_or_srt,
    generate_karaoke_ass,
    export_subtitles_bundle,
    burn_karaoke_to_video,
    COLOR_PALETTES,
)
from ffmpeg_processing import (
    FFMPEG_PATH,
    extract_vertical_clip_9_16,
    get_video_resolution,
    get_video_duration,
)


def create_synthetic_test_video(file_path: str, duration: float = 6.0, fps: int = 30):
    w, h = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    total_frames = int(duration * fps)
    for f in range(total_frames):
        t = f / fps
        # Background color wave
        b = int(128 + 120 * np.sin(t * 2))
        g = int(128 + 120 * np.sin(t * 3 + 1))
        r = int(128 + 120 * np.sin(t * 1.5 + 2))
        frame = np.full((h, w, 3), (b, g, r), dtype=np.uint8)
        cv2.putText(frame, f"Frame {f} | {t:.2f}s", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        out.write(frame)
    out.release()


def run_tests():
    print("=" * 60)
    print("🧪 RUNNING LYRICS & KARAOKE TEST SUITE")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix="test_karaoke_")

    try:
        # 1. Test Word & Phrase Grouping
        print("\n1. Testing Word & Phrase Grouping...")
        sample_words = [
            TimedWord("Baby", 0.5, 0.9),
            TimedWord("you're", 0.9, 1.2),
            TimedWord("a", 1.2, 1.4),
            TimedWord("firework", 1.4, 2.0),
            TimedWord("come", 2.5, 2.8),
            TimedWord("on", 2.8, 3.0),
            TimedWord("let", 3.0, 3.2),
            TimedWord("your", 3.2, 3.4),
            TimedWord("colors", 3.4, 3.8),
            TimedWord("burst", 3.8, 4.3),
        ]
        phrases = group_words_into_phrases(sample_words, max_words_per_line=4)
        assert len(phrases) >= 2, f"Expected at least 2 phrases, got {len(phrases)}"
        print(f"   ✓ Created {len(phrases)} phrases from {len(sample_words)} words")
        for p in phrases:
            print(f"     - [{p.start:.2f}s -> {p.end:.2f}s] {p.text}")

        # 2. Test User Lyrics Alignment
        print("\n2. Testing User Lyrics Text Alignment...")
        user_lyrics = "Baby you're a firework\nCome on let your colors burst"
        aligned = align_user_lyrics_with_audio(user_lyrics, sample_words, audio_duration=5.0)
        assert len(aligned) == len(sample_words), f"Expected {len(sample_words)} aligned words, got {len(aligned)}"
        print(f"   ✓ Aligned {len(aligned)} user words with timestamps")

        # 3. Test LRC and SRT Parsing
        print("\n3. Testing LRC / SRT Parser...")
        sample_lrc = """[00:01.00]First line of the song
[00:03.50]Second line is louder
[00:06.00]Chorus starts right here"""
        lrc_phrases = parse_lrc_or_srt(sample_lrc)
        assert len(lrc_phrases) == 3, f"Expected 3 LRC lines, got {len(lrc_phrases)}"
        print(f"   ✓ Successfully parsed {len(lrc_phrases)} lines from LRC")

        # 4. Test ASS Generation with Multiple Styles & Palettes
        print("\n4. Testing ASS Generation (TikTok Bounce, Karaoke Sweep, Clean Pop)...")
        for style in ["tiktok_bounce", "karaoke_sweep", "clean_pop"]:
            for palette in ["tiktok_yellow", "neon_cyan", "hot_pink"]:
                ass_file = os.path.join(temp_dir, f"test_{style}_{palette}.ass")
                generate_karaoke_ass(
                    phrases=phrases,
                    output_ass_path=ass_file,
                    video_width=1080,
                    video_height=1920,
                    animation_style=style,
                    palette_key=palette,
                    position_mode="bottom",
                    uppercase=True,
                )
                assert os.path.exists(ass_file) and os.path.getsize(ass_file) > 100
        print("   ✓ All ASS style & palette variations generated successfully")

        # 5. Test Subtitle Bundle Export
        print("\n5. Testing Subtitle Bundle Export (.ass, .srt, .lrc)...")
        bundle = export_subtitles_bundle(phrases, "bundle_test", temp_dir)
        assert os.path.exists(bundle["ass"]) and os.path.exists(bundle["srt"]) and os.path.exists(bundle["lrc"])
        print(f"   ✓ Exported bundle: ASS ({os.path.getsize(bundle['ass'])}B), SRT ({os.path.getsize(bundle['srt'])}B), LRC ({os.path.getsize(bundle['lrc'])}B)")

        # 6. Test Video Burning with FFmpeg (16:9 Video)
        print("\n6. Testing Video Burning on 16:9 Video...")
        synth_vid = os.path.join(temp_dir, "synth_16x9.mp4")
        create_synthetic_test_video(synth_vid, duration=5.0)

        ass_16x9 = os.path.join(temp_dir, "test_16x9.ass")
        generate_karaoke_ass(
            phrases=phrases,
            output_ass_path=ass_16x9,
            video_width=640,
            video_height=360,
            animation_style="tiktok_bounce",
            palette_key="tiktok_yellow",
        )

        burned_16x9 = os.path.join(temp_dir, "burned_16x9.mp4")
        ok, err = burn_karaoke_to_video(synth_vid, ass_16x9, burned_16x9, use_nvenc=False)
        assert ok, f"Burn 16:9 failed: {err}"
        assert os.path.exists(burned_16x9) and os.path.getsize(burned_16x9) > 0
        w_b, h_b = get_video_resolution(burned_16x9)
        print(f"   ✓ Burned 16:9 Video: {burned_16x9} | Resolution: {w_b}x{h_b}")

        # 7. Test Vertical 9:16 Extraction with Karaoke Subtitles
        print("\n7. Testing Vertical 9:16 Video Extraction with Baked Karaoke Subtitles...")
        vertical_out = os.path.join(temp_dir, "burned_vertical_9x16.mp4")
        ass_vertical = os.path.join(temp_dir, "test_vertical.ass")
        generate_karaoke_ass(
            phrases=phrases,
            output_ass_path=ass_vertical,
            video_width=1080,
            video_height=1920,
            animation_style="tiktok_bounce",
            palette_key="neon_cyan",
        )

        ok_v, err_v = extract_vertical_clip_9_16(
            video_file=synth_vid,
            start_time=0.5,
            duration=3.5,
            output_file=vertical_out,
            framing_mode="blur_pad",
            target_width=1080,
            target_height=1920,
            fps=30.0,
            use_nvenc=False,
            ass_subtitle_file=ass_vertical,
        )
        assert ok_v, f"Vertical extraction failed: {err_v}"
        assert os.path.exists(vertical_out) and os.path.getsize(vertical_out) > 0
        vw, vh = get_video_resolution(vertical_out)
        assert (vw, vh) == (1080, 1920), f"Expected (1080, 1920), got ({vw}, {vh})"
        print(f"   ✓ Vertical 9:16 Karaoke Clip: {vertical_out} | Resolution: {vw}x{vh}")

        print("\n" + "=" * 60)
        print("🎉 ALL KARAOKE & LYRICS TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)

    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
