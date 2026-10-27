from types import SimpleNamespace

import pytest

from spoof.data import COLUMNS
from spoof.sets import protocol


@pytest.fixture
def paths(tmp_path):
    itw = tmp_path / "itw"
    itw.mkdir()
    (itw / "meta.csv").write_text(
        "file,speaker,label\n0.wav,A,spoof\n1.wav,A,bona-fide\n"
    )
    stage2 = tmp_path / "stage2"
    stage2.mkdir()
    (stage2 / "ruls.csv").write_text(
        "file_name,label,attack,speaker,language,silence,gain_db\n"
        "ruls/a.flac,bonafide,-,5397,ru,orig,1.5\n"
    )
    return SimpleNamespace(asvspoof=None, itw=itw, etrim=None, stage2=stage2)


def test_protocol_trim40(paths):
    orig = protocol("itw", paths)
    trimmed = protocol("itw_trim40", paths)
    assert list(trimmed.columns) == COLUMNS
    assert set(trimmed["silence"]) == {"trim40"}
    assert trimmed.drop(columns="silence").equals(orig.drop(columns="silence"))


def test_protocol_local(paths):
    df = protocol("ruls_trim40", paths)
    assert list(df["file_name"]) == [str(paths.stage2 / "ruls" / "a.flac")]
    assert list(df["silence"]) == ["trim40"]
    assert list(df["gain_db"]) == [1.5]


def test_protocol_unknown(paths):
    with pytest.raises(ValueError):
        protocol("ruls_trim", paths)
