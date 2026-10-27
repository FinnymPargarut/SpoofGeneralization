import pandas as pd
import pytest

from spoof.data import (
    COLUMNS,
    asvspoof2019,
    read_asvspoof2019,
    read_etrim,
    read_in_the_wild,
    read_local,
    read_protocol,
    with_pad,
    with_trim,
)

MARKS = pd.DataFrame({"name": ["a", "b"], "start": [480, 0], "end": [22800, 40640]})


def write(path, text):
    path.write_text(text)
    return path


def test_read_asvspoof2019(tmp_path):
    protocol = write(
        tmp_path / "p.txt",
        "LA_0079 LA_T_1138215 - - bonafide\nLA_0079 LA_T_1271820 - A01 spoof\n",
    )
    df = read_asvspoof2019(protocol, tmp_path / "flac")
    assert list(df.columns) == COLUMNS
    assert df.to_dict("records") == [
        {
            "file_name": str(tmp_path / "flac" / "LA_T_1138215.flac"),
            "label": "bonafide",
            "attack": "-",
            "speaker": "LA_0079",
            "language": "en",
            "silence": "orig",
        },
        {
            "file_name": str(tmp_path / "flac" / "LA_T_1271820.flac"),
            "label": "spoof",
            "attack": "A01",
            "speaker": "LA_0079",
            "language": "en",
            "silence": "orig",
        },
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "LA_0079 LA_T_1138215 - bonafide\n",
        "LA_0079 LA_T_1138215 - - fake\n",
    ],
)
def test_read_asvspoof2019_broken(tmp_path, text):
    with pytest.raises(ValueError):
        read_asvspoof2019(write(tmp_path / "p.txt", text), tmp_path)


def test_read_in_the_wild(tmp_path):
    meta = write(
        tmp_path / "meta.csv",
        "file,speaker,label\n0.wav,Alec Guinness,spoof\n1.wav,Alec Guinness,bona-fide\n",
    )
    df = read_in_the_wild(meta, tmp_path)
    assert list(df.columns) == COLUMNS
    assert list(df["file_name"]) == [str(tmp_path / "0.wav"), str(tmp_path / "1.wav")]
    assert list(df["label"]) == ["spoof", "bonafide"]
    assert set(df["attack"]) == {"-"}
    assert set(df["speaker"]) == {"Alec Guinness"}
    assert set(df["language"]) == {"en"}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "file,speaker\n0.wav,Alec Guinness\n",
        "file,speaker,label\n0.wav,Alec Guinness\n",
        "file,speaker,label\n0.wav,Alec Guinness,real\n",
    ],
)
def test_read_in_the_wild_broken(tmp_path, text):
    with pytest.raises(ValueError):
        read_in_the_wild(write(tmp_path / "meta.csv", text), tmp_path)


def test_protocol_roundtrip(tmp_path):
    protocol = write(tmp_path / "p.txt", "LA_0079 LA_T_1271820 - A01 spoof\n")
    df = read_asvspoof2019(protocol, tmp_path)
    df.to_csv(tmp_path / "protocol.csv", index=False)
    assert read_protocol(tmp_path / "protocol.csv").equals(df)


@pytest.mark.parametrize(
    "text",
    [
        "file_name,label\n/a.flac,spoof\n",
        "file_name,label,attack,speaker,language,silence\n",
        "file_name,label,attack,speaker,language,silence\n/a.flac,fake,-,LA_0079,en,orig\n",
    ],
)
def test_read_protocol_broken(tmp_path, text):
    with pytest.raises(ValueError):
        read_protocol(write(tmp_path / "protocol.csv", text))


def test_read_etrim(tmp_path):
    log = write(
        tmp_path / "trim.log",
        "TRIAL START_INDEX END_INDEX\nLA_D_1000265 480 22800\nLA_D_1003673 0 40640\n",
    )
    assert read_etrim(log).to_dict("records") == [
        {"name": "LA_D_1000265", "start": 480, "end": 22800},
        {"name": "LA_D_1003673", "start": 0, "end": 40640},
    ]


@pytest.mark.parametrize(
    "rows",
    [
        "",
        "LA_D_1000265 480\n",
        "LA_D_1000265 abc 22800\n",
        "LA_D_1000265 500 500\n",
    ],
)
def test_read_etrim_broken(tmp_path, rows):
    log = write(tmp_path / "trim.log", "TRIAL START_INDEX END_INDEX\n" + rows)
    with pytest.raises(ValueError):
        read_etrim(log)


@pytest.mark.parametrize(("split", "kind"), [("train", "trn"), ("eval", "trl")])
def test_asvspoof2019_layout(tmp_path, split, kind):
    protocols = tmp_path / "ASVspoof2019_LA_cm_protocols"
    protocols.mkdir()
    write(
        protocols / f"ASVspoof2019.LA.cm.{split}.{kind}.txt",
        "LA_0079 LA_T_1138215 - - bonafide\n",
    )
    flac = tmp_path / f"ASVspoof2019_LA_{split}" / "flac" / "LA_T_1138215.flac"
    assert list(asvspoof2019(tmp_path, split)["file_name"]) == [str(flac)]


def protocol_of(names):
    return pd.DataFrame(
        {
            "file_name": [f"/la/{name}.flac" for name in names],
            "label": "spoof",
            "attack": "A01",
            "speaker": "LA_0079",
            "language": "en",
            "silence": "orig",
        }
    )


def test_with_trim():
    df = with_trim(protocol_of(["b", "a"]), MARKS)
    assert df[["start", "end"]].to_numpy().tolist() == [[0, 40640], [480, 22800]]
    assert set(df["silence"]) == {"trim"}


def test_with_trim_missing_marks():
    with pytest.raises(ValueError):
        with_trim(protocol_of(["a", "c"]), MARKS)


def test_with_pad():
    donors = with_trim(protocol_of(["a", "b"]), MARKS)
    protocol = protocol_of(list("cdefghijkl"))
    padded = with_pad(protocol, donors, seed=0)
    assert padded.equals(with_pad(protocol, donors, seed=0))
    assert set(padded["silence"]) == {"pad"}
    picks = set(zip(padded["pad_file"], padded["pad_len"]))
    assert picks == {("/la/a.flac", 480), ("/la/b.flac", 0)}


def test_with_pad_without_donors():
    donors = with_trim(protocol_of(["a"]), MARKS).iloc[:0]
    with pytest.raises(ValueError):
        with_pad(protocol_of(["c"]), donors, seed=0)


LOCAL = (
    "file_name,label,attack,speaker,language,silence,gain_db\n"
    "ruls/a.flac,bonafide,-,5397,ru,orig,3.5\n"
)


def test_read_local(tmp_path):
    df = read_local(write(tmp_path / "ruls.csv", LOCAL), tmp_path)
    assert list(df.columns) == [*COLUMNS, "gain_db"]
    assert list(df["file_name"]) == [str(tmp_path / "ruls" / "a.flac")]
    assert df["speaker"].iloc[0] == "5397"
    assert df["gain_db"].iloc[0] == 3.5


@pytest.mark.parametrize(
    "text",
    [
        "",
        "file_name,label,attack,speaker,language,silence\n",
        "label,attack,speaker,language,silence\nbonafide,-,1,ru,orig\n",
        "file_name,label,attack,speaker,language,silence\na.flac,real,-,1,ru,orig\n",
        "file_name,label,attack,speaker,language,silence\na.flac,bonafide,-,,ru,orig\n",
    ],
)
def test_read_local_broken(tmp_path, text):
    with pytest.raises(ValueError):
        read_local(write(tmp_path / "set.csv", text), tmp_path)
