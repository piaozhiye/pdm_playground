#!/usr/bin/env python3
"""Tests for wav2pdm and pdm2pcm closed loop."""
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
WAV2PDM = os.path.join(HERE, "wav2pdm")
PDM2PCM = os.path.join(HERE, "pdm2pcm")


def run(cmd, stdin=None, stdout=None):
    with open(stdin, "rb") if stdin else open(os.devnull, "rb") as fin, \
         open(stdout, "wb") if stdout else open(os.devnull, "wb") as fout:
        p = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
    return p.returncode, p.stderr.decode("utf-8", "replace")


def write_wav(path, sr=8000, n=8000, channels=1, sampwidth=2, freq=440.0):
    """Write mono/stereo S16 sine WAV."""
    frames = []
    for i in range(n):
        v = int(0.5 * 32767 * math.sin(2 * math.pi * freq * i / sr))
        if channels == 1:
            frames.append(struct.pack("<h", v))
        else:
            frames.append(struct.pack("<hh", v, v))
    with wave.open(path, "w") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(b"".join(frames))


def expect_fail(args, stdin_path, label):
    rc, err = run([WAV2PDM] + args, stdin=stdin_path)
    assert rc == 1, f"{label}: expected exit code 1, got {rc}"
    assert err.strip(), f"{label}: expected stderr message"
    print(f"OK reject: {label}")


def test_cli_rejects():
    tmp = tempfile.mkdtemp(prefix="wav2pdm_")
    good = os.path.join(tmp, "good8k.wav")
    write_wav(good, sr=8000, n=100)

    # missing -f
    expect_fail(["-d", "128"], good, "missing -f")
    # missing -d
    expect_fail(["-f", "1024000"], good, "missing -d")
    # bad d
    expect_fail(["-f", "1024000", "-d", "32"], good, "d not 64/128")
    # f % d != 0
    expect_fail(["-f", "1000000", "-d", "128"], good, "f not divisible by d")
    # wrong sample rate (44.1k vs f/d=8k)
    bad_sr = os.path.join(tmp, "bad44k.wav")
    write_wav(bad_sr, sr=44100, n=100)
    expect_fail(["-f", "1024000", "-d", "128"], bad_sr, "sr mismatch")
    # stereo
    st = os.path.join(tmp, "st.wav")
    write_wav(st, sr=8000, n=100, channels=2)
    rc, err = run([WAV2PDM, "-f", "1024000", "-d", "128"], stdin=st, stdout=os.path.join(tmp, "st.dat"))
    assert rc == 0, f"stereo should be accepted, rc={rc} err={err}"
    print("OK accept: valid 8k stereo")

    # IEEE float (format tag 3): patch fmt tag on a valid PCM WAV
    flt = os.path.join(tmp, "float.wav")
    write_wav(flt, sr=8000, n=100)
    with open(flt, "r+b") as f:
        f.seek(20)  # RIFF(12) + "fmt "(4) + chunk size(4) → format tag
        f.write(struct.pack("<H", 3))
    expect_fail(["-f", "1024000", "-d", "128"], flt, "float wav")

    # valid input produces non-empty output
    out = os.path.join(tmp, "out.dat")
    rc, err = run([WAV2PDM, "-f", "1024000", "-d", "128"], stdin=good, stdout=out)
    assert rc == 0, f"valid wav should succeed, rc={rc} err={err}"
    assert os.path.getsize(out) > 0, "valid wav should produce bytes"
    print("OK accept: valid 8k mono")


def pearson(a, b):
    n = min(len(a), len(b))
    if n < 100:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def read_s16le(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 2
    return list(struct.unpack("<%dh" % n, data[: n * 2]))


def read_wav_payload(path):
    """Read PCM sample payload only (skip 44-byte RIFF header)."""
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2, "expected S16 WAV"
        frames = w.readframes(w.getnframes())
    n = len(frames) // 2
    return list(struct.unpack("<%dh" % n, frames[: n * 2]))


def best_lag_pearson(a, b, max_lag=200):
    """Return (best_rho, best_lag) maximizing Pearson ρ over lag ∈ [-max_lag, max_lag].

    Positive lag means b is delayed relative to a (decoder group delay).
    Lag-max measures alignment/goodness of match only; on periodic signals the
    signed max relocates to the opposite lobe under inversion, so polarity is
    asserted separately via zero-lag rho0 > 0.
    """
    best_rho, best_lag = -2.0, 0
    n = len(a)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x = a[: n - lag]
            y = b[lag:lag + len(x)]
        else:
            y = b[: n + lag]
            x = a[-lag:-lag + len(y)]
        rho = pearson(x, y)
        if rho > best_rho:
            best_rho, best_lag = rho, lag
    return best_rho, best_lag


def roundtrip(f, d, sr=8000, sec=1.0, freq=440.0, warmup_ms=50, max_lag=200):
    tmp = tempfile.mkdtemp(prefix="rt_")
    wav = os.path.join(tmp, "in.wav")
    pdm = os.path.join(tmp, "out.pdm")
    raw = os.path.join(tmp, "dec.raw")
    n = int(sr * sec)
    write_wav(wav, sr=sr, n=n, freq=freq)

    rc, err = run([WAV2PDM, "-f", str(f), "-d", str(d)], stdin=wav, stdout=pdm)
    assert rc == 0, f"wav2pdm failed: {err}"
    assert os.path.getsize(pdm) > 0

    rc, err = run([PDM2PCM, "-f", str(f), "-d", str(d)], stdin=pdm, stdout=raw)
    assert rc == 0, f"pdm2pcm failed: {err}"

    orig = read_wav_payload(wav)
    dec = read_s16le(raw)
    skip = sr * warmup_ms // 1000
    # trim possible ~1ms EOF garbage from decoder
    usable = min(len(orig), len(dec)) - 8
    a = orig[skip:usable]
    b = dec[skip:usable]
    rho0 = pearson(a, b)
    rho, lag = best_lag_pearson(a, b, max_lag=max_lag)
    print(
        f"roundtrip f={f} d={d}: len_orig={len(orig)} len_dec={len(dec)} "
        f"rho0={rho0:.4f} best_rho={rho:.4f} lag={lag}"
    )
    # polarity: zero-lag correlation only (lag-max is polarity-blind on periodic sines)
    assert rho0 > 0, f"zero-lag polarity inverted: rho0={rho0}"
    assert abs(rho0) > 0.1, f"zero-lag correlation not meaningful: rho0={rho0}"
    assert abs(rho) > 0.9, f"correlation too low: {rho} (zero-lag rho0={rho0})"
    assert rho > 0, f"polarity inverted: {rho}"
    # decoder group delay (HP/LP + sinc^3) is finite and positive, not wild
    assert 0 <= lag <= max_lag, f"implausible lag: {lag}"
    # rough frequency: zero-crossing count on decoded AC signal
    # count rising zero crossings in middle 0.5s
    mid = b[len(b) // 4: 3 * len(b) // 4]
    crossings = sum(1 for i in range(1, len(mid)) if mid[i - 1] <= 0 < mid[i])
    # duration of mid at sr
    dur = len(mid) / sr
    est_f = crossings / dur if dur > 0 else 0
    # allow 15% (HP/LP + sigma-delta)
    assert abs(est_f - freq) / freq < 0.15, f"freq est {est_f} vs {freq}"


def test_pdm2pcm_c_rejects():
    import subprocess
    tmpdir = tempfile.mkdtemp(prefix="pdm2pcm_c_")
    pcm = os.path.join(tmpdir, "out.raw")
    cases = [
        ("-c 3",  ["pdm2pcm", "-f", "1024000", "-d", "128", "-c", "3"]),
        ("-c 9",  ["pdm2pcm", "-f", "1024000", "-d", "128", "-c", "9"]),
        ("-c 0",  ["pdm2pcm", "-f", "1024000", "-d", "128", "-c", "0"]),
    ]
    with open(os.devnull, "rb") as fin, open(pcm, "wb") as fout:
        for label, cmd in cases:
            full = [PDM2PCM] + cmd[1:]
            r = subprocess.run(full, stdin=fin, stdout=fout, stderr=subprocess.PIPE)
            assert r.returncode == 1, f"{label}: expected rc=1, got {r.returncode}"
            assert r.stderr.decode().strip(), f"{label}: expected stderr message"
            print(f"OK reject pdm2pcm: {label}")


def write_stereo_wav(path, sr=8000, n=8000, fL=440.0, fR=880.0, amp=0.5):
    """Write a stereo S16 WAV with different sines per channel."""
    import math, struct, wave
    frames = []
    for i in range(n):
        vL = int(amp * 32767 * math.sin(2 * math.pi * fL * i / sr))
        vR = int(amp * 32767 * math.sin(2 * math.pi * fR * i / sr))
        frames.append(struct.pack("<hh", vL, vR))
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(frames))


def split_stereo(raw_samples):
    """Split L R L R ... into ([L0,L1,...], [R0,R1,...])."""
    L = raw_samples[0::2]
    R = raw_samples[1::2]
    return L, R


def roundtrip_stereo(f, d, sr=8000, sec=1.0, fL=440.0, fR=880.0, warmup_ms=50):
    import subprocess, tempfile, os
    tmp = tempfile.mkdtemp(prefix="rt_st_")
    wav = os.path.join(tmp, "in.wav")
    pdm = os.path.join(tmp, "out.pdm")
    raw = os.path.join(tmp, "dec.raw")
    n = int(sr * sec)
    write_stereo_wav(wav, sr=sr, n=n, fL=fL, fR=fR)

    r = subprocess.run([WAV2PDM, "-f", str(f), "-d", str(d)],
                       stdin=open(wav, "rb"), stdout=open(pdm, "wb"),
                       stderr=subprocess.PIPE)
    assert r.returncode == 0, f"wav2pdm failed: {r.stderr.decode()}"
    assert os.path.getsize(pdm) > 0, "empty PDM"

    r = subprocess.run([PDM2PCM, "-f", str(f), "-d", str(d), "-c", "2"],
                       stdin=open(pdm, "rb"), stdout=open(raw, "wb"),
                       stderr=subprocess.PIPE)
    assert r.returncode == 0, f"pdm2pcm failed: {r.stderr.decode()}"

    orig_L, orig_R = split_stereo(read_wav_payload(wav))
    dec = read_s16le(raw)
    dec_L, dec_R = split_stereo(dec)

    skip = sr * warmup_ms // 1000
    usable = min(len(orig_L), len(dec_L)) - 8
    assert usable > skip + 100, f"too few decoded samples: orig={len(orig_L)} dec={len(dec_L)}"

    # Per-channel lag-tolerant correlation; both channels independently.
    for label, orig_ch, dec_ch, freq in (("L", orig_L, dec_L, fL), ("R", orig_R, dec_R, fR)):
        a = orig_ch[skip:usable]
        b = dec_ch[skip:usable]
        rho0 = pearson(a, b)
        best_rho, best_lag = best_lag_pearson(a, b, max_lag=200)
        print(f"  stereo {label} (f={freq}): rho0={rho0:.4f} best_rho={best_rho:.4f} lag={best_lag}")
        assert abs(rho0) > 0.1, f"stereo {label} zero-lag ρ too low: {rho0}"
        assert rho0 > 0, f"stereo {label} polarity inverted: {rho0}"
        assert best_rho > 0.9, f"stereo {label} best ρ too low: {best_rho}"
        # Frequency sanity on decoded channel: zero-crossing count near expected freq.
        mid = b[len(b)//4 : 3*len(b)//4]
        crossings = sum(1 for i in range(1, len(mid)) if mid[i-1] <= 0 < mid[i])
        est_f = crossings / (len(mid) / sr) if mid else 0
        assert abs(est_f - freq) / freq < 0.15, f"stereo {label} freq est {est_f} vs {freq}"


def test_roundtrip_stereo_128():
    roundtrip_stereo(1024000, 128)


def test_roundtrip_stereo_64():
    roundtrip_stereo(512000, 64)


def test_roundtrip_128():
    roundtrip(1024000, 128)


def test_roundtrip_64():
    roundtrip(512000, 64)


def test_stereo_silence_bitpattern():
    """Stereo silence at d=64 should produce the exact 16-byte pattern 95 95 55 55 …."""
    import math, wave, struct, subprocess, tempfile, os
    tmp = tempfile.mkdtemp()
    src = os.path.join(tmp, "st.wav")
    # 1 stereo frame at sr=16000 (= 1 PCM sample per channel) needs only 4 bytes of data
    # f=1024000, d=64 → sr_required = f/d = 16000
    with wave.open(src, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x00\x00\x00\x00")  # L=0, R=0, 1 frame
    out = os.path.join(tmp, "st.pdm")
    r = subprocess.run([WAV2PDM, "-f", "1024000", "-d", "64"], stdin=open(src,"rb"), stdout=open(out,"wb"), stderr=subprocess.PIPE)
    assert r.returncode == 0, f"wav2pdm stereo failed rc={r.returncode} stderr={r.stderr.decode()}"
    data = open(out, "rb").read()
    expected = bytes.fromhex("95955555555555555555555555555555")
    assert len(data) == len(expected), f"len mismatch {len(data)} vs {len(expected)}"
    assert data == expected, f"byte mismatch:\n  got: {data.hex(' ')}\n  exp: {expected.hex(' ')}"
    print(f"OK pack stereo: {len(data)} bytes = {data.hex()}")


def test_pack_bitpattern():
    """Exact packed bytes for all-zero input (Σ-Δ + MSB-first pack)."""
    tmp = tempfile.mkdtemp(prefix="pack_")
    wav = os.path.join(tmp, "zero.wav")
    pdm = os.path.join(tmp, "zero.pdm")
    n, d = 4, 64
    write_wav(wav, sr=8000, n=n, freq=0.0)  # sine(0) → all samples 0

    rc, err = run([WAV2PDM, "-f", "512000", "-d", str(d)], stdin=wav, stdout=pdm)
    assert rc == 0, f"wav2pdm failed: {err}"

    # length = n_samples * d / 8 (d ∈ {64,128} always byte-aligned)
    expected_len = (n * d + 7) // 8
    assert os.path.getsize(pdm) == expected_len, (
        f"size {os.path.getsize(pdm)} != {expected_len}"
    )

    # Hand-computed Σ-Δ bits for x=0, acc=0:
    #   1,0,0,1,0,1,0,1 → 0x95 (first byte, MSB = oldest bit)
    # then acc drifts but pattern settles to 0,1,0,1,... → 0x55 each byte
    expected = bytes([0x95] + [0x55] * (expected_len - 1))
    with open(pdm, "rb") as f:
        got = f.read()
    assert got == expected, f"pack mismatch:\n got={got.hex()}\n exp={expected.hex()}"
    print(f"OK pack: {expected_len} bytes = {got.hex()}")


def main():
    assert os.path.isfile(WAV2PDM) and os.access(WAV2PDM, os.X_OK), "build wav2pdm first"
    assert os.path.isfile(PDM2PCM) and os.access(PDM2PCM, os.X_OK), "build pdm2pcm first"
    test_cli_rejects()
    test_pdm2pcm_c_rejects()
    test_stereo_silence_bitpattern()
    test_pack_bitpattern()
    test_roundtrip_128()
    test_roundtrip_64()
    test_roundtrip_stereo_128()
    test_roundtrip_stereo_64()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
