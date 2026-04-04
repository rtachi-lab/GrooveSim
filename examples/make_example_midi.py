from __future__ import annotations

from pathlib import Path

import mido


def main() -> None:
    out = Path("examples") / "simple_pattern.mid"
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("program_change", program=0, time=0))

    durations = [480, 480, 240, 240, 480, 480, 240, 240]
    notes = [36, 42, 38, 42, 36, 42, 38, 46]

    for note, delta in zip(notes, durations):
        track.append(mido.Message("note_on", note=note, velocity=100, time=0))
        track.append(mido.Message("note_off", note=note, velocity=0, time=delta))

    mid.save(out)
    print(out)


if __name__ == "__main__":
    main()
