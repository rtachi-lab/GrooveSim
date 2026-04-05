from __future__ import annotations

import argparse
import json

from .midi import render_midi_to_wav_file
from .pipeline import analyze_audio_file, analyze_midi_file, analyze_onset_file, save_json


def add_tempo_prior_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tempo-min", dest="tempo_min", type=float, help="Optional lower bound for tempo estimation in BPM.")
    parser.add_argument("--tempo-max", dest="tempo_max", type=float, help="Optional upper bound for tempo estimation in BPM.")


def get_tempo_prior(args: argparse.Namespace) -> tuple[float, float] | None:
    tempo_min = getattr(args, "tempo_min", None)
    tempo_max = getattr(args, "tempo_max", None)
    if tempo_min is None and tempo_max is None:
        return None
    if tempo_min is None or tempo_max is None:
        raise SystemExit("Both --tempo-min and --tempo-max must be provided together.")
    return (float(tempo_min), float(tempo_max))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate groove-related features from audio or onset sequences.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio_parser = subparsers.add_parser("analyze-audio", help="Analyze an audio file.")
    audio_parser.add_argument("path", help="Path to audio file.")
    audio_parser.add_argument("--json-out", dest="json_out", help="Optional output path for JSON results.")
    add_tempo_prior_arguments(audio_parser)

    midi_parser = subparsers.add_parser("analyze-midi", help="Render a MIDI file to audio and analyze it.")
    midi_parser.add_argument("path", help="Path to MIDI file.")
    midi_parser.add_argument("--json-out", dest="json_out", help="Optional output path for JSON results.")
    add_tempo_prior_arguments(midi_parser)
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
    tempo_prior = get_tempo_prior(args)

    if args.command == "analyze-audio":
        result = analyze_audio_file(args.path, tempo_prior=tempo_prior)
    elif args.command == "analyze-midi":
        result = analyze_midi_file(args.path, tempo_prior=tempo_prior)
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
