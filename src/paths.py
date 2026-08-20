#!/usr/bin/env python3
"""Shared project paths and directory helpers."""

import os

from logger import ROOT_DIR


INPUT_DIR = os.path.join(ROOT_DIR, 'input')
AUDIO_INPUT_DIR = os.path.join(INPUT_DIR, 'audio')
VIDEO_INPUT_DIR = os.path.join(INPUT_DIR, 'video')
PROCESSING_DIR = os.path.join(INPUT_DIR, 'processing')
GRADIO_TEMP_DIR = os.path.join(INPUT_DIR, 'gradio_uploads')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
SHORTS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'tiktok_shorts')
SUBTITLES_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'subtitles')


def ensure_project_dirs() -> None:
    """Create the standard project directories if they are missing."""
    for directory in [
        INPUT_DIR,
        AUDIO_INPUT_DIR,
        VIDEO_INPUT_DIR,
        PROCESSING_DIR,
        GRADIO_TEMP_DIR,
        OUTPUT_DIR,
        SHORTS_OUTPUT_DIR,
        SUBTITLES_OUTPUT_DIR,
    ]:
        os.makedirs(directory, exist_ok=True)


def get_input_dir() -> str:
    """Get the local input directory path."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    return INPUT_DIR


def get_audio_input_dir() -> str:
    """Get the audio input directory path."""
    os.makedirs(AUDIO_INPUT_DIR, exist_ok=True)
    return AUDIO_INPUT_DIR


def get_video_input_dir() -> str:
    """Get the video input directory path."""
    os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)
    return VIDEO_INPUT_DIR


def get_processing_dir() -> str:
    """Get the processing directory path."""
    os.makedirs(PROCESSING_DIR, exist_ok=True)
    return PROCESSING_DIR


def get_gradio_temp_dir() -> str:
    """Get the Gradio upload/temp directory path."""
    os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
    return GRADIO_TEMP_DIR


def get_output_dir() -> str:
    """Get the final output directory path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def get_shorts_output_dir() -> str:
    """Get the TikTok/Shorts output directory path."""
    os.makedirs(SHORTS_OUTPUT_DIR, exist_ok=True)
    return SHORTS_OUTPUT_DIR


def get_subtitles_output_dir() -> str:
    """Get the subtitles output directory path."""
    os.makedirs(SUBTITLES_OUTPUT_DIR, exist_ok=True)
    return SUBTITLES_OUTPUT_DIR


ensure_project_dirs()
