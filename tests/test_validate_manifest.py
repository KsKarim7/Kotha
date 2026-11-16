import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from validate_manifest import main
from wer import MANIFEST_COLUMNS

TEXT_MARKER = "গোপন বাক্য SECRET"


def write_audio(
    path: Path, frames: int = 16_000, sample_rate: int = 16_000, channels: int = 1
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros((frames, channels), np.float32), sample_rate, format=sf_format(path))
    return path


def sf_format(path: Path) -> str:
    return {".wav": "WAV", ".flac": "FLAC"}[path.suffix]


def manifest_row(row_id: str, path: str, **overrides: str) -> dict[str, str]:
    row = {
        "id": row_id,
        "path": path,
        "speaker": "spk01",
        "lang": "bn",
        "noise": "clean",
        "split": "dev",
        "text": f"{TEXT_MARKER} {row_id}",
    }
    return {**row, **overrides}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(
    capsys: pytest.CaptureFixture[str], manifests: list[Path], data_dir: Path, *options: str
) -> tuple[int, list[str]]:
    exit_code = main(["--manifest", *map(str, manifests), "--data-dir", str(data_dir), *options])
    output = capsys.readouterr().out
    assert TEXT_MARKER not in output
    assert "SECRET" not in output
    return exit_code, output.splitlines()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    audio_dir = tmp_path / "asr"
    write_audio(audio_dir / "a.wav")
    return audio_dir


def test_a_valid_public_and_private_set_passes_with_the_exact_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audio_dir = tmp_path / "asr"
    write_audio(audio_dir / "spk01" / "a.wav", frames=3 * 16_000)
    write_audio(audio_dir / "spk02" / "b.wav", frames=6 * 16_000)
    write_audio(audio_dir / "spk03" / "c.wav", frames=9 * 16_000)
    public = write_manifest(
        tmp_path / "public.csv",
        [
            manifest_row("k1", "spk01/a.wav"),
            manifest_row(
                "k2", "spk02/b.wav", speaker="spk02", lang="mix", noise="street", split="test"
            ),
        ],
    )
    private = write_manifest(
        tmp_path / "private.csv",
        [manifest_row("k3", "spk03/c.wav", speaker="spk03", lang="en", noise="fan")],
    )

    exit_code, lines = run(capsys, [public, private], audio_dir)

    assert exit_code == 0
    assert lines == [
        "3 rows, 3 speakers, 0.30 minutes",
        "",
        "lang x noise (clips / minutes)",
        "lang             clean        street           fan",
        "bn            1 / 0.05      0 / 0.00      0 / 0.00",
        "en            0 / 0.00      0 / 0.00      1 / 0.15",
        "mix           0 / 0.00      1 / 0.10      0 / 0.00",
        "",
        "speaker   clips  minutes",
        "spk01         1     0.05",
        "spk02         1     0.10",
        "spk03         1     0.15",
        "",
        "split     clips  minutes",
        "dev           2     0.20",
        "test          1     0.10",
        "",
        "codecs: WAV/PCM_16 (3)",
        "sample rates: 16000 Hz (3)",
        "",
        "0 errors, 0 warnings",
    ]


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"lang": ""}, "m.csv:2 k1: lang must be one of bn, en, mix"),
        ({"lang": "all"}, "m.csv:2 k1: lang must be one of bn, en, mix"),
        ({"noise": "rain"}, "m.csv:2 k1: noise must be one of clean, street, fan"),
        ({"split": "train"}, "m.csv:2 k1: split must be one of dev, test"),
        ({"speaker": "spk1"}, "m.csv:2 k1: speaker must be spk followed by two digits"),
        ({"speaker": "spk001"}, "m.csv:2 k1: speaker must be spk followed by two digits"),
        ({"speaker": "SPK01"}, "m.csv:2 k1: speaker must be spk followed by two digits"),
        ({"speaker": "spk০১"}, "m.csv:2 k1: speaker must be spk followed by two digits"),
        ({"id": ""}, "m.csv:2 -: empty id"),
        ({"text": ""}, "m.csv:2 k1: empty text"),
        ({"path": ""}, "m.csv:2 k1: empty path"),
        ({"path": "/abs/a.wav"}, "m.csv:2 k1: path is absolute"),
        ({"path": "../asr/a.wav"}, "m.csv:2 k1: path contains .."),
        ({"path": "sub/../a.wav"}, "m.csv:2 k1: path contains .."),
        ({"path": "missing.wav"}, "m.csv:2 k1: file missing"),
    ],
)
def test_each_row_error_is_reported_with_its_location(
    tmp_path: Path,
    data_dir: Path,
    capsys: pytest.CaptureFixture[str],
    overrides: dict[str, str],
    expected: str,
) -> None:
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("k1", "a.wav") | overrides])

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert expected in lines
    assert lines[-1].startswith("1 errors")


def test_a_path_that_resolves_outside_the_data_dir_is_an_error(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = write_audio(tmp_path / "outside.wav")
    (data_dir / "link.wav").symlink_to(outside)
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("k1", "link.wav")])

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert "m.csv:2 k1: path resolves outside --data-dir" in lines


def test_repeated_ids_and_paths_point_to_the_first_row(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_audio(data_dir / "b.wav")
    manifest = write_manifest(
        tmp_path / "m.csv",
        [
            manifest_row("k1", "a.wav"),
            manifest_row("k1", "b.wav"),
            manifest_row("k3", "./a.wav"),
        ],
    )

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert lines[:2] == [
        "m.csv:3 k1: id repeated (first at m.csv:2)",
        "m.csv:4 k3: path repeated (first at m.csv:2)",
    ]


def test_an_id_repeated_across_the_two_manifests_is_an_error(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_audio(data_dir / "b.wav")
    public = write_manifest(tmp_path / "public.csv", [manifest_row("k1", "a.wav")])
    private = write_manifest(tmp_path / "private.csv", [manifest_row("k1", "b.wav")])

    exit_code, lines = run(capsys, [public, private], data_dir)

    assert exit_code == 1
    assert lines[0] == "private.csv:2 k1: id repeated (first at public.csv:2)"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("junk.m4a", "m.csv:2 k1: unreadable by soundfile (real AAC .m4a?)"),
        ("junk.aac", "m.csv:2 k1: unreadable by soundfile (real AAC .m4a?)"),
        ("junk.wav", "m.csv:2 k1: unreadable by soundfile"),
    ],
)
def test_a_file_soundfile_cannot_read_is_an_error(
    tmp_path: Path,
    data_dir: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
    expected: str,
) -> None:
    (data_dir / name).write_bytes(b"not audio at all" * 64)
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("k1", name)])

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert lines[0] == expected


@pytest.mark.parametrize("frames", [8_000, 16_000])
def test_durations_exactly_at_the_bounds_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], frames: int
) -> None:
    write_audio(tmp_path / "asr" / "a.wav", frames=frames)
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("k1", "a.wav")])

    exit_code, lines = run(capsys, [manifest], tmp_path / "asr", "--min-s", "0.5", "--max-s", "1")

    assert exit_code == 0
    assert lines[-1] == "0 errors, 0 warnings"


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (7_200, "m.csv:2 k1: duration 0.45 s is outside [0.5, 1.0] s"),
        (16_800, "m.csv:2 k1: duration 1.05 s is outside [0.5, 1.0] s"),
    ],
)
def test_durations_outside_the_bounds_are_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], frames: int, expected: str
) -> None:
    write_audio(tmp_path / "asr" / "a.wav", frames=frames)
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("k1", "a.wav")])

    exit_code, lines = run(capsys, [manifest], tmp_path / "asr", "--min-s", "0.5", "--max-s", "1")

    assert exit_code == 1
    assert lines[0] == expected


def test_a_wav_named_mp3_warns_about_the_container(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (data_dir / "a.mp3").write_bytes((data_dir / "a.wav").read_bytes())
    manifest = write_manifest(tmp_path / "m.csv", [manifest_row("k1", "a.mp3")])

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 0
    assert lines[0] == "m.csv:2 k1: warning: WAV data in a file named .mp3"
    assert lines[-1] == "0 errors, 1 warnings"


def test_low_sample_rates_and_extra_channels_are_warnings(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_audio(data_dir / "low.flac", frames=8_000, sample_rate=8_000)
    write_audio(data_dir / "stereo.wav", channels=2)
    manifest = write_manifest(
        tmp_path / "m.csv", [manifest_row("k1", "low.flac"), manifest_row("k2", "stereo.wav")]
    )

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 0
    assert lines[:4] == [
        "m.csv:2 k1: warning: sample rate 8000 Hz is below 16000 Hz",
        "m.csv:3 k2: warning: 2 channels, downmixed to mono",
        "warning: the set mixes 2 codecs: FLAC/PCM_16 (1), WAV/PCM_16 (1)",
        "warning: the set mixes 2 sample rates: 8000 Hz (1), 16000 Hz (1)",
    ]
    assert "codecs: FLAC/PCM_16 (1), WAV/PCM_16 (1)" in lines
    assert "sample rates: 8000 Hz (1), 16000 Hz (1)" in lines


def test_invalid_values_are_summarised_as_other_without_echoing_them(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = write_manifest(
        tmp_path / "m.csv",
        [manifest_row("k1", "a.wav", lang="SECRET-lang", speaker="SECRET-spk", split="all")],
    )

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert "1 rows, 0 speakers, 0.02 minutes" in lines
    assert "other         1 / 0.02      0 / 0.00      0 / 0.00" in lines
    assert "other         1     0.02" in lines


def test_a_short_row_reports_its_empty_fields(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"{','.join(MANIFEST_COLUMNS)}\nk1,a.wav\n", encoding="utf-8")

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert lines[:5] == [
        "m.csv:2 k1: empty text",
        "m.csv:2 k1: lang must be one of bn, en, mix",
        "m.csv:2 k1: noise must be one of clean, street, fan",
        "m.csv:2 k1: split must be one of dev, test",
        "m.csv:2 k1: speaker must be spk followed by two digits",
    ]


def test_a_manifest_without_the_required_columns_is_an_error(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "bad.csv"
    manifest.write_text("id,path\nk1,a.wav\n", encoding="utf-8")

    exit_code, lines = run(capsys, [manifest], data_dir)

    assert exit_code == 1
    assert lines[0].startswith("bad.csv: manifest ")
    assert lines[0].endswith("is missing columns: speaker, lang, noise, split, text")
    assert lines[-1] == "1 errors, 0 warnings"
