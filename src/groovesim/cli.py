from __future__ import annotations

import argparse
import json

from .midi import render_midi_to_wav_file
from .pipeline import analyze_audio_file, analyze_midi_file, analyze_onset_file, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate groove-related features from audio or onset sequences.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio_parser = subparsers.add_parser("analyze-audio", help="Analyze an audio file.")
    audio_parser.add_argument("path", help="Path to audio file.")
    audio_parser.add_argument("--json-out", dest="json_out", help="Optional output path for JSON results.")

    midi_parser = subparsers.add_parser("analyze-midi", help="Render a MIDI file to audio and analyze it.")
    midi_parser.add_argument("path", help="Path to MIDI file.")
    midi_parser.add_argument("--json-out", dest="json_out", help="Optional output path for JSON results.")
    midi_parser.add_argument(
        "--rendered-wav-out",
        dest="rendered_wav_out",
        help="Optional path to save the rendered MIDI audio as WAV.",
    )

    onset_parser = subparsers.add_parser("analyze-onsets", help="Analyze onset times from JSON or CSV.")
    onset_parser.add_argument("path", help="Path to onset JSON or CSV file.")
    onset_parser.add_argument("--json-out", dest="json_out", help="Optional output path for JSON results.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "analyze-audio":
        result = analyze_audio_file(args.path)
    elif args.command == "analyze-midi":
        result = analyze_midi_file(args.path)
        if args.rendered_wav_out:
            render_midi_to_wav_file(args.path, output_path=args.rendered_wav_out)
    elif args.command == "analyze-onsets":
        result = analyze_onset_file(args.path)
    else:
        parser.error(f"Unsupported command: {args.command}")
        return

    if args.json_out:
        save_json(result, args.json_out)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
