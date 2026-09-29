#!/usr/bin/env python3
# Copyright (c) 2026 piaozhiye <piaozhiye@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Tests for wav2pdm and pdm2pcm closed loop."""
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_pdm_24bit as A24

HERE = os.path.dirname(os.path.abspath(__file__))
WAV2PDM = os.path.join(HERE, "wav2pdm")
PDM2PCM = os.path.join(HERE, "pdm2pcm")
ANALYZE_PDM = os.path.join(HERE, "analyze_pdm.py")
ANALYZE_24BIT = os.path.join(HERE, "analyze_pdm_24bit.py")


def run(cmd, stdin=None, stdout=None):
    with open(stdin, "rb") if stdin else open(os.devnull, "rb") as fin, \
         open(stdout, "wb") if stdout else open(os.devnull, "wb") as fout:
        p = subprocess.run(
            cmd,
            stdin=fin,
            stdout=fout,
            stderr=subprocess.PIPE,
            timeout=10,
        )
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


CHECK_DOC_NUMBERS = os.path.join(HERE, "check_doc_numbers.py")
REF_DECIMATOR = os.path.join(HERE, "pdm_ref_decimator")


def non_sinusoidal_residual(signal, bits, freq, sample_rate, skip=24000, n=49152):
    """Residual after projecting the known fundamental out of `signal`.

    This is the metric this project settled on, and the reason matters:
    an integer-lag alignment cannot remove a fractional group delay.
    OpenPDMFilter delays by 2977.09 samples, and the leftover 0.09 sample
    is 0.0019 of a period at 1 kHz, which on a 11585 LSB16 signal shows up
    as ~136 LSB16 of pure phase residue. That artifact was large enough to
    hide every real bit-depth effect.

    Fitting [1, cos, sin] at the known frequency absorbs gain, phase
    (fractional delay) and DC at once, with no lag search. The constant
    column must be inside the basis: an earlier version subtracted the
    window mean and fit only [cos, sin], which over a window that is not a
    whole number of tone periods couples the fundamental into the residual
    DC (~25 LSB16 of leakage at n=8192, 170.67 periods). With DC in the
    basis the tone is removed exactly for ANY window length, and the floor
    stops depending on where the window starts or how long it is.

    What is left is exactly what we care about: harmonics, aliasing and
    quantization noise.

    Returns the residual expressed in 16-bit LSB regardless of `bits`, so
    a 16-bit output (native int16 units) and a 24-bit output (native int32
    units divided by 256) are directly comparable.
    """
    skip = min(skip, max(0, len(signal) - n - 1))
    n = min(n, len(signal) - skip)
    assert n > 1024, "not enough samples"
    x = np.asarray(signal[skip:skip + n], dtype=np.float64)
    t = np.arange(n, dtype=np.float64)
    omega = 2.0 * math.pi * freq / sample_rate
    basis = np.column_stack((np.ones(n), np.cos(omega * t), np.sin(omega * t)))
    coef, *_ = np.linalg.lstsq(basis, x, rcond=None)
    residual = x - basis @ coef
    amplitude = math.hypot(coef[1], coef[2])
    rms = math.sqrt(float(np.mean(residual ** 2)))
    # int16 samples are already in 16-bit LSB; int32 samples are 256x finer.
    if bits == 24:
        rms /= 256.0
    return {
        "rms_16lsb": rms,
        "amplitude": amplitude,
        "snr_db": 20 * math.log10(amplitude / rms) if rms > 0 else float("inf"),
    }


def run_pdm2pcm_fullscale(bits, pattern, out_path, frames=4000):
    """Drive pdm2pcm with a saturated PDM density and return the raw output.

    pattern 0 is an all-ones density (+full scale), 1 is all-zeros
    (-full scale), 2 is a deterministic sweep. A constant density only
    reaches the filter's steady state, so pattern 3 alternates all-ones and
    all-zeros block by block: that is the largest possible swing and the
    only pattern that actually drives the 24-bit quantizer to its rail.
    """
    pdm_path = out_path + ".pdm"
    # pdm2pcm reads one 1 ms block at a time and rejects a partial one.
    # At f=3072000, d=64, mono a block is 3072000/1000 = 3072 bits = 384 bytes.
    block_bytes = 384
    blocks = max(1, frames // block_bytes)
    with open(pdm_path, "wb") as stream:
        for index in range(blocks):
            if pattern == 0:
                stream.write(b"\xff" * block_bytes)
            elif pattern == 1:
                stream.write(b"\x00" * block_bytes)
            elif pattern == 3:
                stream.write((b"\xff" if index % 2 == 0 else b"\x00") * block_bytes)
            else:
                stream.write(bytes((index * 37 + b * 11) & 0xFF for b in range(block_bytes)))
    with open(pdm_path, "rb") as fin, open(out_path, "wb") as fout:
        p = subprocess.run(
            [
                PDM2PCM,
                "-f", "3072000",
                "-d", "64",
                "-c", "1",
                "-b", str(bits),
            ],
            stdin=fin,
            stdout=fout,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    return out_path


def test_pdm2pcm_24bit_fullscale_saturates():
    """The 24-bit path must clamp to the 24-bit range at full scale.

    Without saturation the int32 cast in open_pdm_quantize_24 would wrap
    around, so a full-scale input could come out as a small sample with the
    wrong sign instead of a clipped one.

    A constant density is not enough to reach the rail: the filter settles
    at a fixed operating point. The alternating pattern does reach it, so
    the assertion below is only meaningful if that pattern is included.
    """
    with tempfile.TemporaryDirectory(prefix="pdm24_clip_") as tmp:
        # Each path clips to its own bound: 24-bit to the full int24 range,
        # 16-bit to +-32700 (deliberately below +-32767 to leave headroom).
        for bits, reader, low, high, rail_low, rail_high in (
            (24, read_s24le, -0x800000, 0x7FFFFF, -0x800000, 0x7FFFFF),
            (16, read_s16le, -32700, 32700, -32700, 32700),
        ):
            for pattern, name in ((0, "all-ones"), (1, "all-zeros"),
                                  (3, "alternating")):
                raw = run_pdm2pcm_fullscale(
                    bits, pattern, os.path.join(tmp, f"p{pattern}_{bits}.raw")
                )
                samples = reader(raw)
                assert samples, f"{bits}-bit {name} produced no samples"
                assert all(low <= value <= high for value in samples), (
                    f"{bits}-bit {name} escaped the {low}..{high} range"
                )
            # Guard the premise: the alternating pattern must actually clip,
            # otherwise this test could pass with saturation removed.
            raw = run_pdm2pcm_fullscale(
                bits, 3, os.path.join(tmp, f"rail{bits}.raw")
            )
            samples = reader(raw)
            assert min(samples) == rail_low or max(samples) == rail_high, (
                f"{bits}-bit alternating pattern never reached the rail "
                f"(range {min(samples)}..{max(samples)}), so the saturation "
                f"assertion above is vacuous"
            )
    print("OK 24-bit full-scale stays inside the 24-bit range")


def write_raw_wav(path, sr, frames, channels, bits, block_align, freq=440.0):
    """Write a PCM WAV with an explicit block_align, including wrong ones.

    Used to build headers that claim a frame width inconsistent with
    channels x bits/8 so the parser can be tested against them.
    """
    width = bits // 8
    payload = bytearray()
    for n in range(frames):
        value = int(0.5 * ((1 << (bits - 1)) - 1) * math.sin(2 * math.pi * freq * n / sr))
        for _ in range(channels):
            unsigned = value & ((1 << bits) - 1)
            payload.extend(unsigned.to_bytes(width, "little"))
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sr,
        sr * channels * (bits // 8),
        block_align,
        bits,
    )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    data_chunk = b"data" + struct.pack("<I", len(payload)) + bytes(payload)
    body = b"WAVE" + fmt_chunk + data_chunk
    with open(path, "wb") as stream:
        stream.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def write_s24_wav(path, sr=8000, n=8000, channels=1, freq=440.0):
    """Write mono/stereo packed signed 24-bit little-endian WAV."""
    frames = []
    for i in range(n):
        v = int(0.5 * 8388607 * math.sin(2 * math.pi * freq * i / sr))
        raw = v & 0xFFFFFF
        sample = bytes((raw & 0xFF, (raw >> 8) & 0xFF, (raw >> 16) & 0xFF))
        frames.append(sample * channels)
    with wave.open(path, "w") as w:
        w.setnchannels(channels)
        w.setsampwidth(3)
        w.setframerate(sr)
        w.writeframes(b"".join(frames))


def expect_fail(args, stdin_path, label):
    rc, err = run([WAV2PDM] + args, stdin=stdin_path)
    assert rc == 1, f"{label}: expected exit code 1, got {rc}"
    assert err.strip(), f"{label}: expected stderr message"
    print(f"OK reject: {label}")


def test_cli_rejects():
    with tempfile.TemporaryDirectory(prefix="wav2pdm_") as tmp:
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
        rc, err = run(
            [WAV2PDM, "-f", "1024000", "-d", "128"],
            stdin=st,
            stdout=os.path.join(tmp, "st.dat"),
        )
        assert rc == 0, f"stereo should be accepted, rc={rc} err={err}"
        print("OK accept: valid 8k stereo")

        # numeric options must be parsed completely
        expect_fail(["-f", "1024000junk", "-d", "128"], good, "trailing -f garbage")
        expect_fail(["-f", " 1024000", "-d", "128"], good, "leading whitespace -f")
        expect_fail(["-f", "+1024000", "-d", "128"], good, "leading plus -f")
        expect_fail(
            ["-f", "1024000", "-d", "128", "-o", "1junk"],
            good,
            "trailing -o garbage",
        )
        expect_fail(
            ["-f", "1024000", "-d", "128", "unexpected"],
            good,
            "unexpected positional operand",
        )

        # IEEE float (format tag 3): patch fmt tag on a valid PCM WAV
        flt = os.path.join(tmp, "float.wav")
        write_wav(flt, sr=8000, n=100)
        with open(flt, "r+b") as f:
            f.seek(20)  # RIFF(12) + "fmt "(4) + chunk size(4) → format tag
            f.write(struct.pack("<H", 3))
        expect_fail(["-f", "1024000", "-d", "128"], flt, "float wav")

        # block_align (fmt+12) must agree with channels × bits/8. A header
        # claiming the wrong frame width would make us walk the data chunk
        # on the wrong frame boundaries.
        for bits, align, label in (
            (16, 4, "16-bit mono block_align=4 (stereo claim)"),
            (24, 2, "24-bit mono block_align=2 (16-bit frame width)"),
            (24, 6, "24-bit mono block_align=6 (stereo claim)"),
        ):
            bad_align = os.path.join(tmp, f"align{align}_{bits}.wav")
            write_raw_wav(
                bad_align,
                sr=8000,
                frames=64,
                channels=1,
                bits=bits,
                block_align=align,
            )
            expect_fail(["-f", "1024000", "-d", "128"], bad_align, label)

        # A correct block_align must still be accepted.
        for bits, channels in ((16, 1), (16, 2), (24, 1), (24, 2)):
            ok = os.path.join(tmp, f"ok_{bits}_{channels}.wav")
            write_raw_wav(
                ok,
                sr=8000,
                frames=64,
                channels=channels,
                bits=bits,
                block_align=channels * bits // 8,
            )
            rc, err = run(
                [WAV2PDM, "-f", "1024000", "-d", "128"],
                stdin=ok,
                stdout=os.path.join(tmp, "align_ok.dat"),
            )
            assert rc == 0, f"valid block_align rejected for {bits}-bit/{channels}ch: {err}"

        # valid input produces non-empty output
        out = os.path.join(tmp, "out.dat")
        rc, err = run([WAV2PDM, "-f", "1024000", "-d", "128"], stdin=good, stdout=out)
        assert rc == 0, f"valid wav should succeed, rc={rc} err={err}"
        assert os.path.getsize(out) > 0, "valid wav should produce bytes"
        print("OK accept: valid 8k mono")


def test_wav_odd_chunk_padding():
    """An odd-sized fmt chunk must consume its RIFF pad byte."""
    with tempfile.TemporaryDirectory(prefix="wav_odd_") as tmp:
        wav = os.path.join(tmp, "odd.wav")
        pdm = os.path.join(tmp, "odd.pdm")
        fmt = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16) + b"\x00"
        data = struct.pack("<h", 0)
        fmt_chunk = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"\x00"
        data_chunk = b"data" + struct.pack("<I", len(data)) + data
        payload = b"WAVE" + fmt_chunk + data_chunk
        with open(wav, "wb") as stream:
            stream.write(b"RIFF" + struct.pack("<I", len(payload)) + payload)
        rc, err = run([WAV2PDM, "-f", "512000", "-d", "64"], stdin=wav, stdout=pdm)
        assert rc == 0, err
        assert os.path.getsize(pdm) == 8
        print("OK WAV odd-chunk padding")


def test_wav2pdm_accepts_s24_source():
    """wav2pdm must encode a packed signed 24-bit WAV at the right scale.

    Acceptance alone is not enough: if the modulator kept the 16-bit
    thresholds, a 24-bit sample would slam the loop and the decoded signal
    would be full-scale chatter instead of the source sine.
    """
    sr, n, freq = 8000, 8000, 440.0
    source = [int(0.5 * 8388607 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    with tempfile.TemporaryDirectory(prefix="wav_s24_") as tmp:
        wav = os.path.join(tmp, "source24.wav")
        pdm = os.path.join(tmp, "source24.pdm")
        raw = os.path.join(tmp, "source24.raw")
        write_s24_wav(wav, sr=sr, n=n, freq=freq)
        rc, err = run([WAV2PDM, "-f", "512000", "-d", "64"], stdin=wav, stdout=pdm)
        assert rc == 0, err
        assert os.path.getsize(pdm) == n * 512000 // sr // 8

        rc, err = run([PDM2PCM, "-f", "512000", "-d", "64", "-b", "24"], stdin=pdm, stdout=raw)
        assert rc == 0, err
        decoded = read_s24le(raw)
        assert len(decoded) == n

        skip = sr // 10
        best_rho, best_lag = best_lag_pearson(source[skip:], decoded[skip:], max_lag=400)
        src_peak = max(abs(value) for value in source[skip:])
        dec_peak = max(abs(value) for value in decoded[skip:])
        print(
            f"  s24 source roundtrip: best_rho={best_rho:.4f} lag={best_lag} "
            f"peak_ratio={dec_peak / src_peak:.4f}"
        )
        assert best_rho > 0.9, f"24-bit source best rho too low: {best_rho}"
        assert 0.5 < dec_peak / src_peak < 1.6, f"24-bit source scale wrong: {dec_peak / src_peak}"
        print("OK wav2pdm accepts 24-bit source")


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


def read_s24le(path):
    """Read packed signed 24-bit little-endian samples."""
    with open(path, "rb") as f:
        data = f.read()
    assert len(data) % 3 == 0, len(data)
    values = []
    for i in range(0, len(data), 3):
        value = data[i] | (data[i + 1] << 8) | (data[i + 2] << 16)
        if value & 0x800000:
            value -= 0x1000000
        values.append(value)
    return values


def test_pdm2pcm_24bit_output():
    """The -b 24 option emits packed signed 24-bit little-endian PCM."""
    with tempfile.TemporaryDirectory(prefix="pdm_24_") as tmp:
        wav = os.path.join(tmp, "input.wav")
        pdm = os.path.join(tmp, "input.pdm")
        raw16 = os.path.join(tmp, "output16.raw")
        raw24 = os.path.join(tmp, "output24.raw")
        write_wav(wav, sr=8000, n=8000, freq=440.0)

        rc, err = run(
            [WAV2PDM, "-f", "1024000", "-d", "128"],
            stdin=wav,
            stdout=pdm,
        )
        assert rc == 0, err

        rc, err = run(
            [PDM2PCM, "-f", "1024000", "-d", "128", "-b", "16"],
            stdin=pdm,
            stdout=raw16,
        )
        assert rc == 0, err

        rc, err = run(
            [PDM2PCM, "-f", "1024000", "-d", "128", "-b", "24"],
            stdin=pdm,
            stdout=raw24,
        )
        assert rc == 0, err
        assert os.path.getsize(raw24) == 8000 * 3

        samples16 = read_s16le(raw16)
        samples = read_s24le(raw24)
        assert len(samples) == 8000
        assert len(samples16) == 8000
        assert max(abs(sample) for sample in samples) > 1000
        assert all(-0x800000 <= sample <= 0x7FFFFF for sample in samples)
        assert any(sample & 0xFF for sample in samples), "24-bit output has no LSB detail"
        normalized = [sample / 256.0 for sample in samples]
        denominator = sum(value * value for value in normalized)
        scale = sum(a * b for a, b in zip(samples16, normalized)) / denominator
        assert abs(scale - 1.0) < 0.02, f"24-bit scale mismatch: {scale}"

        with open(os.devnull, "rb") as fin, open(os.path.join(tmp, "bad.raw"), "wb") as fout:
            p = subprocess.run(
                [PDM2PCM, "-f", "1024000", "-d", "128", "-b", "32"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=1,
            )
        assert p.returncode == 1
        assert "16 or 24" in p.stderr.decode("utf-8", "replace")
        print("OK pdm2pcm 24-bit output")


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


def roundtrip(f, d, sr=8000, sec=1.0, freq=440.0, warmup_ms=50, max_lag=200, order=1):
    with tempfile.TemporaryDirectory(prefix="rt_") as tmp:
        wav = os.path.join(tmp, "in.wav")
        pdm = os.path.join(tmp, "out.pdm")
        raw = os.path.join(tmp, "dec.raw")
        n = int(sr * sec)
        write_wav(wav, sr=sr, n=n, freq=freq)

        rc, err = run(
            [WAV2PDM, "-f", str(f), "-d", str(d), "-o", str(order)],
            stdin=wav,
            stdout=pdm,
        )
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
            f"roundtrip order={order} f={f} d={d}: "
            f"len_orig={len(orig)} len_dec={len(dec)} "
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


def test_pdm2pcm_rejects_invalid_buffer_geometry():
    """Reject rates that cannot produce complete 1 ms decimation blocks."""
    cases = [
        (["-f", "64", "-d", "64"], "zero-byte buffer"),
        (["-f", "102400", "-d", "128"], "fractional 1 ms block"),
        (["-f", "1000000", "-d", "128"], "frequency not aligned to block"),
    ]
    with open(os.devnull, "rb") as fin:
        for args, label in cases:
            with tempfile.TemporaryDirectory(prefix="pdm_geometry_") as tmp:
                out_path = os.path.join(tmp, "out.raw")
                with open(out_path, "wb") as fout:
                    p = subprocess.run(
                        [PDM2PCM] + args,
                        stdin=fin,
                        stdout=fout,
                        stderr=subprocess.PIPE,
                        timeout=1,
                    )
                assert p.returncode == 1, (
                    f"{label}: expected rc=1, got {p.returncode}; "
                    f"stderr={p.stderr.decode('utf-8', 'replace')}"
                )
                assert p.stderr.decode("utf-8", "replace").strip(), label
                assert os.path.getsize(out_path) == 0, label
                print(f"OK reject pdm2pcm geometry: {label}")


def test_pdm2pcm_rejects_unsupported_pcm_rate():
    """Reject PCM rates that cannot be represented by the filter API."""
    with tempfile.TemporaryDirectory(prefix="pdm_rate_") as tmp:
        output_path = os.path.join(tmp, "output.raw")
        with open(os.devnull, "rb") as fin, open(output_path, "wb") as fout:
            p = subprocess.run(
                [PDM2PCM, "-f", "4194304000", "-d", "64"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=2,
            )
        assert p.returncode == 1, p.returncode
        assert "PCM sampling rate" in p.stderr.decode("utf-8", "replace")
        assert os.path.getsize(output_path) == 0
        print("OK reject pdm2pcm unsupported PCM rate")


def test_pdm2pcm_rejects_partial_block():
    """Emit complete blocks, but reject a trailing partial block."""
    with tempfile.TemporaryDirectory(prefix="pdm_partial_") as tmp:
        input_path = os.path.join(tmp, "input.pdm")
        output_path = os.path.join(tmp, "output.raw")
        # f=512000, d=64, mono => 512 PDM bits = 64 bytes per 1 ms block.
        with open(input_path, "wb") as stream:
            stream.write(b"\x00" * 65)
        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            p = subprocess.run(
                [PDM2PCM, "-f", "512000", "-d", "64"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=1,
            )
        assert p.returncode == 1, p.returncode
        assert "Incomplete PDM block" in p.stderr.decode("utf-8", "replace")
        # The complete first block is valid and is emitted before the error.
        assert os.path.getsize(output_path) == 16
        print("OK reject pdm2pcm partial block")


def test_order2_fullscale_ubsan():
    """Run second-order encoding with UBSan on a full-scale stress input."""
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        print("SKIP order2 UBSan: no C compiler")
        return
    with tempfile.TemporaryDirectory(prefix="wav2pdm_ubsan_") as tmp:
        binary = os.path.join(tmp, "wav2pdm-ubsan")
        wav_path = os.path.join(tmp, "fullscale.wav")
        pdm_path = os.path.join(tmp, "fullscale.pdm")
        compile_cmd = [
            compiler,
            "-std=gnu99",
            "-Wall",
            "-Wextra",
            "-O1",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-I",
            HERE,
            os.path.join(HERE, "wav2pdm.c"),
            "-o",
            binary,
        ]
        compiled = subprocess.run(
            compile_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
        assert compiled.returncode == 0, compiled.stderr.decode("utf-8", "replace")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(struct.pack("<h", 32767) * 4000)
        with open(wav_path, "rb") as fin, open(pdm_path, "wb") as fout:
            p = subprocess.run(
                [binary, "-f", "512000", "-d", "64", "-o", "2"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=5,
            )
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        print("OK order2 full-scale under UBSan")


def test_24bit_decode_fullscale_ubsan():
    """Sweep the 24-bit decode path at full scale under UBSan.

    The 24-bit quantizer multiplies the filter state by 256 before dividing
    and saturates at +-8388608, so its intermediates are 256x the 16-bit
    path. The 16-bit full-scale guard does not cover it, and the second
    order accumulator state already reaches 27% of STATE_LIMIT at 24 bits.
    The CIC9 variant (CicOrder=9) is swept too: its partial sums reach
    2^54 at d=64, the d=128 kernel relies on the init-time tap rescaling,
    and the block-alternating pattern drives the quantizer to both rails,
    so an accumulator wrap would show up here as undefined behavior.
    """
    compiler = shutil.which("gcc") or shutil.which("cc")
    if compiler is None:
        print("SKIP 24-bit UBSan: no C compiler")
        return
    driver = """
#include <stdio.h>
#include <stdint.h>
#include "OpenPDMFilter.h"

static void run_24(int order, int vol, int pattern, int cic) {
  TPDMFilter_InitStruct f;
  static uint8_t pdm[96];
  static int32_t out24[192];
  f.MaxVolume = (uint8_t)vol;
  f.In_MicChannels = 1; f.Out_MicChannels = 1;
  f.CicOrder = (uint8_t)cic;
  f.Decimation = (uint8_t)order; f.Fs = 48000; f.LP_HZ = 15000; f.HP_HZ = 10;
  f.nSamples = (order == 64) ? 48 : 96;
  Open_PDM_Filter_Init(&f);
  for (int i = 0; i < 2000; i++) {
    int ns = (order == 64) ? 48 : 96;
    for (int b = 0; b < ns; b++) {
      if (pattern == 0) pdm[b] = 0xFF;
      else if (pattern == 1) pdm[b] = 0x00;
      else if (pattern == 3) pdm[b] = ((i / ns) % 2) ? 0x00 : 0xFF;
      else pdm[b] = (uint8_t)(i * 37 + b * 11);
    }
    if (order == 64) Open_PDM_Filter_64_24(pdm, out24, 16, &f);
    else             Open_PDM_Filter_128_24(pdm, out24, 16, &f);
  }
}

int main(void) {
  for (int cic = 3; cic <= 9; cic += 6)
    for (int pattern = 0; pattern <= 3; pattern++)
      for (int vol = 1; vol <= 16; vol++) {
        run_24(64, vol, pattern, cic);
        run_24(128, vol, pattern, cic);
      }
  printf("24-bit full-scale sweep clean\\n");
  return 0;
}
"""
    with tempfile.TemporaryDirectory(prefix="pdm24_ubsan_") as tmp:
        driver_path = os.path.join(tmp, "driver.c")
        binary = os.path.join(tmp, "filter24-ubsan")
        with open(driver_path, "w") as stream:
            stream.write(driver)
        compiled = subprocess.run(
            [
                compiler,
                "-std=gnu99",
                "-Wall",
                "-Wextra",
                "-O1",
                "-fsanitize=undefined",
                "-fno-sanitize-recover=undefined",
                "-I",
                HERE,
                driver_path,
                os.path.join(HERE, "OpenPDMFilter.c"),
                "-o",
                binary,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        assert compiled.returncode == 0, compiled.stderr.decode("utf-8", "replace")
        p = subprocess.run(
            [binary],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        assert b"clean" in p.stdout, p.stdout.decode("utf-8", "replace")
        print("OK 24-bit decode full-scale under UBSan")


def test_pdm2pcm_cic_n_args():
    """-n 3 is byte-identical to the default; invalid orders are rejected."""
    with tempfile.TemporaryDirectory(prefix="cicn_args_") as tmp:
        pdm = os.path.join(tmp, "s.pdm")
        with open(pdm, "wb") as f:
            f.write(b"\x55" * 384 * 4)   # 4 x 1 ms blocks, f=3072000 d=64 mono
        a = os.path.join(tmp, "a.raw")
        b = os.path.join(tmp, "b.raw")
        rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c1"], stdin=pdm, stdout=a)
        assert rc == 0, err
        rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c1", "-n", "3"], stdin=pdm, stdout=b)
        assert rc == 0, err
        assert open(a, "rb").read() == open(b, "rb").read(), "-n 3 must equal the default"
        for bad in ("4", "0", "abc", "9x"):
            rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c1", "-n", bad], stdin=pdm)
            assert rc == 1, (bad, rc, err)
            assert "3 or 9" in err, (bad, err)
        rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c1", "-n", "9"],
                      stdin=pdm, stdout=os.path.join(tmp, "c.raw"))
        assert rc == 0, err
        print("OK pdm2pcm -n argument handling")


def test_pdm2pcm_cic9_fullscale_bounds():
    """Full-scale drive through -n 9 -b 24 stays at the rails, never wraps.

    The N=9 accumulator reaches 2^53 at d=64 (Hogenauer W2) and the d=128
    kernel relies on the init-time tap rescaling; a width regression shows
    up as sign-flipped garbage instead of clean rails.
    """
    with tempfile.TemporaryDirectory(prefix="cic9_fs_") as tmp:
        pdm = os.path.join(tmp, "s.pdm")
        blocks = 8
        with open(pdm, "wb") as f:
            for index in range(blocks):
                f.write((b"\xff" if index % 2 == 0 else b"\x00") * 384)
        out = os.path.join(tmp, "s.raw")
        rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c1", "-b24", "-n", "9"],
                      stdin=pdm, stdout=out)
        assert rc == 0, err
        raw = open(out, "rb").read()
        assert len(raw) == 48 * blocks * 3, len(raw)
        vals = []
        for i in range(0, len(raw), 3):
            v = raw[i] | (raw[i + 1] << 8) | (raw[i + 2] << 16)
            vals.append(v - 0x1000000 if v & 0x800000 else v)
        assert all(-8388608 <= v <= 8388607 for v in vals)
        assert min(vals) <= -8388600 and max(vals) >= 8388600, (min(vals), max(vals))
        print("OK CIC9 full-scale stays at the rails")


def test_pdm2pcm_cic9_advantage():
    """-n 9 lowers the chain floor under the 16-bit grid at d=128.

    Anchors: the CIC3 floors are the published study numbers (24-bit source,
    0.5 amplitude, order 2, 1 kHz). At d=128 the CIC9 24-bit floor drops to
    ~0.07 LSB16, below the 16-bit grid (0.289), so the 16-bit container
    becomes the bottleneck and the closed-form model
    sqrt(e24^2 + 0.289^2)/e24 predicts the measured 16b/24b ratio.
    """
    with tempfile.TemporaryDirectory(prefix="cic9_adv_") as tmp:
        wav = os.path.join(tmp, "src.wav")
        A24.write_stereo_wav_bits(wav, sample_rate=48000, frames=96000,
                                  left_hz=1000.0, right_hz=3000.0,
                                  amplitude=0.5, bits=24)
        floors = {}
        for d, rate in ((64, 3072000), (128, 6144000)):
            pdm = os.path.join(tmp, f"s{d}.pdm")
            rc, err = run([WAV2PDM, "-f", str(rate), "-d", str(d), "-o2"],
                          stdin=wav, stdout=pdm)
            assert rc == 0, err
            for order in (3, 9):
                for bits in (16, 24):
                    out = os.path.join(tmp, f"n{order}d{d}b{bits}.raw")
                    rc, err = run([PDM2PCM, "-f", str(rate), "-d", str(d),
                                   "-c2", "-b", str(bits), "-n", str(order)],
                                  stdin=pdm, stdout=out)
                    assert rc == 0, err
                    left, _ = A24.read_channels(out, bits)
                    floors[(order, d, bits)] = non_sinusoidal_residual(
                        left, bits, 1000.0, 48000)["rms_16lsb"]
        # anchors: the stock N=3 numbers from the bit-depth study
        for d, e16, e24 in ((64, 1.4174, 1.3886), (128, 0.3925, 0.2545)):
            assert abs(floors[(3, d, 16)] - e16) / e16 < 0.05, (d, floors[(3, d, 16)])
            assert abs(floors[(3, d, 24)] - e24) / e24 < 0.05, (d, floors[(3, d, 24)])
        # CIC9 d=128: floor below the 16-bit grid -> real 24-bit benefit.
        # e24 is pinned to a narrow band around the known value: a mutation
        # that hardcodes the stage count back to 3 (or otherwise degrades
        # the kernel) lands far outside this band even when it barely moves
        # the other assertions.
        e24 = floors[(9, 128, 24)]
        e16 = floors[(9, 128, 16)]
        assert 0.05 < e24 < 0.12, e24
        adv = e16 / e24
        pred = math.sqrt(e24 ** 2 + 0.289 ** 2) / e24
        assert adv > 3.0, adv
        assert abs(adv - pred) / pred < 0.10, (adv, pred)
        # CIC9 d=64: better kernel, but the floor stays above the grid
        assert floors[(9, 64, 24)] < floors[(3, 64, 24)]
        adv64 = floors[(9, 64, 16)] / floors[(9, 64, 24)]
        assert 1.0 < adv64 < 1.6, adv64
        # the stock N=3 d=64 advantage stays negligible
        assert floors[(3, 64, 16)] / floors[(3, 64, 24)] < 1.05
        # -n 9 must produce a genuinely different stream than -n 3
        assert open(os.path.join(tmp, "n9d128b24.raw"), "rb").read() != \
               open(os.path.join(tmp, "n3d128b24.raw"), "rb").read(), \
               "-n 9 output must differ from -n 3"
        print("OK CIC9 advantage: d=128 %.2fx (%.1f dB), d=64 %.2fx"
              % (adv, 20 * math.log10(adv), adv64))


def test_order_option_changes_pdm_stream():
    """The explicit -o option must select a different deterministic stream."""
    with tempfile.TemporaryDirectory(prefix="pdm_order_") as tmp:
        wav = os.path.join(tmp, "input.wav")
        first = os.path.join(tmp, "order1.pdm")
        second = os.path.join(tmp, "order2.pdm")
        write_wav(wav, sr=8000, n=100, freq=440.0)
        for order, output in ((1, first), (2, second)):
            rc, err = run(
                [WAV2PDM, "-f", "512000", "-d", "64", "-o", str(order)],
                stdin=wav,
                stdout=output,
            )
            assert rc == 0, err
        with open(first, "rb") as stream:
            first_data = stream.read()
        with open(second, "rb") as stream:
            second_data = stream.read()
        assert first_data
        assert second_data
        assert first_data != second_data, "-o 1 and -o 2 produced identical PDM"
        print("OK order option changes PDM stream")


def test_pdm2pcm_c_rejects():
    import subprocess
    with tempfile.TemporaryDirectory(prefix="pdm2pcm_c_") as tmpdir:
        pcm = os.path.join(tmpdir, "out.raw")
        cases = [
            ("-c 3",  ["pdm2pcm", "-f", "1024000", "-d", "128", "-c", "3"]),
            ("-c 9",  ["pdm2pcm", "-f", "1024000", "-d", "128", "-c", "9"]),
            ("-c 0",  ["pdm2pcm", "-f", "1024000", "-d", "128", "-c", "0"]),
        ]
        with open(os.devnull, "rb") as fin, open(pcm, "wb") as fout:
            for label, cmd in cases:
                full = [PDM2PCM] + cmd[1:]
                r = subprocess.run(
                    full,
                    stdin=fin,
                    stdout=fout,
                    stderr=subprocess.PIPE,
                    timeout=10,
                )
                assert r.returncode == 1, f"{label}: expected rc=1, got {r.returncode}"
                assert r.stderr.decode().strip(), f"{label}: expected stderr message"
                print(f"OK reject pdm2pcm: {label}")


def test_pdm2pcm_strict_arguments():
    cases = [
        (["-f", " 1024000", "-d", "128"], "leading whitespace -f"),
        (["-f", "+1024000", "-d", "128"], "leading plus -f"),
        (["-f", "1024000", "-d", "128", "unexpected"], "unexpected operand"),
    ]
    with tempfile.TemporaryDirectory(prefix="pdm_args_") as tmp:
        for args, label in cases:
            output_path = os.path.join(tmp, "output.raw")
            with open(os.devnull, "rb") as fin, open(output_path, "wb") as fout:
                p = subprocess.run(
                    [PDM2PCM] + args,
                    stdin=fin,
                    stdout=fout,
                    stderr=subprocess.PIPE,
                    timeout=1,
                )
            assert p.returncode == 1, f"{label}: expected rc=1, got {p.returncode}"
            assert p.stderr.decode("utf-8", "replace").strip(), label
            print(f"OK reject pdm2pcm strict argument: {label}")


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


def roundtrip_stereo(
    f, d, sr=8000, sec=1.0, fL=440.0, fR=880.0, warmup_ms=50, order=1
):
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory(prefix="rt_st_") as tmp:
        wav = os.path.join(tmp, "in.wav")
        pdm = os.path.join(tmp, "out.pdm")
        raw = os.path.join(tmp, "dec.raw")
        n = int(sr * sec)
        write_stereo_wav(wav, sr=sr, n=n, fL=fL, fR=fR)

        with open(wav, "rb") as fin, open(pdm, "wb") as fout:
            r = subprocess.run(
                [WAV2PDM, "-f", str(f), "-d", str(d), "-o", str(order)],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        assert r.returncode == 0, f"wav2pdm failed: {r.stderr.decode()}"
        assert os.path.getsize(pdm) > 0, "empty PDM"

        with open(pdm, "rb") as fin, open(raw, "wb") as fout:
            r = subprocess.run(
                [PDM2PCM, "-f", str(f), "-d", str(d), "-c", "2"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=10,
            )
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
            print(
                f"  stereo order={order} {label} (f={freq}): rho0={rho0:.4f} "
                f"best_rho={best_rho:.4f} lag={best_lag}"
            )
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
    """Stereo silence at d=64 should produce the exact 16-byte pattern."""
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory(prefix="pack_stereo_") as tmp:
        src = os.path.join(tmp, "st.wav")
        out = os.path.join(tmp, "st.pdm")
        # 1 stereo frame at sr=16000 (= 1 PCM sample per channel)
        with wave.open(src, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"\x00\x00\x00\x00")
        with open(src, "rb") as fin, open(out, "wb") as fout:
            r = subprocess.run(
                [WAV2PDM, "-f", "1024000", "-d", "64"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                timeout=10,
            )
        assert r.returncode == 0, (
            f"wav2pdm stereo failed rc={r.returncode} stderr={r.stderr.decode()}"
        )
        with open(out, "rb") as stream:
            data = stream.read()
        expected = bytes.fromhex("95955555555555555555555555555555")
        assert len(data) == len(expected), f"len mismatch {len(data)} vs {len(expected)}"
        assert data == expected, (
            f"byte mismatch:\n  got: {data.hex(' ')}\n  exp: {expected.hex(' ')}"
        )
        print(f"OK pack stereo: {len(data)} bytes = {data.hex()}")


def test_pack_bitpattern():
    """Exact packed bytes for all-zero input (Σ-Δ + MSB-first pack)."""
    with tempfile.TemporaryDirectory(prefix="pack_") as tmp:
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


def load_analysis_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze_pdm", ANALYZE_PDM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_script_coherent_measurement_defaults():
    module = load_analysis_module()
    args = module.build_parser().parse_args([])
    assert args.fft_size == 49152
    assert args.left_hz == 1000.0
    assert args.right_hz == 3000.0
    for frequency in (args.left_hz, args.right_hz):
        signal = 0.5 * 32767.0 * module.np.sin(
            2.0 * module.np.pi * frequency * module.np.arange(args.fft_size) / args.sample_rate
        )
        metrics = module.fft_metrics(
            signal,
            args.sample_rate,
            frequency,
            args.fft_size,
            0,
        )
        assert metrics["thdn_db"] < -80.0, (frequency, metrics)
    print("OK analyzer coherent default FFT")


def test_analysis_lag_uses_overlap_correlation():
    module = load_analysis_module()
    rng = module.np.random.default_rng(0)
    reference = rng.normal(size=256)
    decoded = module.np.concatenate([module.np.zeros(37), reference[:-37]])
    lag, correlation = module.best_positive_lag(reference, decoded, 80)
    assert lag == 37
    assert correlation > 0.99, correlation
    print("OK analyzer overlap correlation")


def test_analysis_script_timeout_and_input_validation():
    import importlib.util

    spec = importlib.util.spec_from_file_location("analyze_pdm", ANALYZE_PDM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="analyze_timeout_") as tmp:
        input_path = os.path.join(tmp, "input")
        output_path = os.path.join(tmp, "output")
        with open(input_path, "wb"):
            pass
        try:
            module.run_tool(
                [sys.executable, "-c", "import time; time.sleep(2)"],
                input_path,
                output_path,
                timeout=0.05,
            )
        except RuntimeError as exc:
            assert "timed out" in str(exc).lower()
        else:
            raise AssertionError("hanging analyzer subprocess was not rejected")

        try:
            module.run_tool(
                [sys.executable, "-c", "import sys; sys.exit(7)"],
                input_path,
                output_path,
                timeout=1,
            )
        except RuntimeError as exc:
            assert "failed (7)" in str(exc)
        else:
            raise AssertionError("failed analyzer subprocess was not rejected")

    p = subprocess.run(
        [sys.executable, ANALYZE_PDM, "--seconds", "inf"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    assert p.returncode == 1
    assert "Traceback" not in p.stderr.decode("utf-8", "replace")

    p = subprocess.run(
        [sys.executable, ANALYZE_PDM, "--seconds", "2.0003"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    assert p.returncode == 1
    error = p.stderr.decode("utf-8", "replace")
    assert "whole 1 ms" in error
    assert "Traceback" not in error
    print("OK analyzer timeout and input validation")


def test_analysis_24bit_script_self_test():
    assert os.path.isfile(ANALYZE_24BIT), "analyze_pdm_24bit.py is missing"
    p = subprocess.run(
        [sys.executable, ANALYZE_24BIT, "--self-test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    assert b"self-test passed" in p.stdout.lower(), p.stdout.decode("utf-8", "replace")
    print("OK 24-bit analysis self-test")


def test_non_sinusoidal_residual_metric():
    """Lock the measurement metric: units, invariance, and sensitivity.

    The bug this guards against cost two wrong conclusions: an integer-lag
    alignment cannot remove a fractional group delay, and 2977.09 samples
    of delay leaked 136 LSB16 of pure phase residue into the result.
    """
    sample_rate = 48000
    n = 96000
    t = np.arange(n)
    ref = (0.5 * 8388607 * np.sin(2 * math.pi * 1000.0 * t / sample_rate)).astype(np.float64)

    # A pure tone must leave only its own quantization noise.
    m = non_sinusoidal_residual(ref, 24, 1000.0, sample_rate)
    assert m["rms_16lsb"] < 0.005, f"pure tone leaked: {m['rms_16lsb']}"
    assert m["snr_db"] > 100, m["snr_db"]

    # A fractional delay must NOT create residual. This is the exact failure
    # mode of the integer-lag metric: 0.09 of a sample at 1 kHz is 0.0019
    # of a period and produced 136 LSB16 of phase residue.
    for frac in (0.09, 0.25, 0.5):
        shifted = np.interp(
            t, t - frac, ref, left=ref[0], right=ref[-1]
        )
        m = non_sinusoidal_residual(shifted, 24, 1000.0, sample_rate)
        assert m["rms_16lsb"] < 0.01, (
            f"fractional delay {frac} leaked: {m['rms_16lsb']}"
        )

    # An integer delay must not create residual either.
    shifted = np.concatenate([np.zeros(37), ref[:-37]])
    m = non_sinusoidal_residual(shifted, 24, 1000.0, sample_rate)
    assert m["rms_16lsb"] < 0.01, f"integer delay leaked: {m['rms_16lsb']}"

    # A pure gain change must not create residual.
    m = non_sinusoidal_residual(ref * 0.37, 24, 1000.0, sample_rate)
    assert m["rms_16lsb"] < 0.01, f"gain change leaked: {m['rms_16lsb']}"

    # Injected white noise must show up at the injected scale.
    rng = np.random.default_rng(7)
    noisy = ref + rng.normal(0, 20.0 * 256, n)
    m = non_sinusoidal_residual(noisy, 24, 1000.0, sample_rate)
    assert 15.0 < m["rms_16lsb"] < 25.0, f"noise scale wrong: {m['rms_16lsb']}"

    # 16-bit and 24-bit quantization of the same tone must differ by
    # exactly the 16-bit grid, and both must be reported in LSB16.
    # int16 samples are native 16-bit units; int32 are 256x finer.
    q16_native = np.clip(np.rint(ref / 256.0), -32700, 32700)
    m16 = non_sinusoidal_residual(q16_native, 16, 1000.0, sample_rate)
    m24 = non_sinusoidal_residual(q16_native * 256.0, 24, 1000.0, sample_rate)
    assert 0.2 < m16["rms_16lsb"] < 0.4, (
        f"16-bit grid should be ~0.29 LSB16, got {m16['rms_16lsb']}"
    )
    assert abs(m16["rms_16lsb"] - m24["rms_16lsb"]) < 0.01, (m16, m24)
    print("OK non_sinusoidal_residual metric invariants")


def test_ref_decimator_matches_stock_layout():
    """pdm_ref_decimator must decode the same PDM the same way pdm2pcm does.

    Guards the packed-stereo layout and the bit order. A previous version
    treated the channels as blocked instead of byte interleaved, which
    silently swapped them.
    """
    if not os.path.isfile(REF_DECIMATOR) or not os.access(REF_DECIMATOR, os.X_OK):
        print("SKIP ref decimator: not built")
        return
    with tempfile.TemporaryDirectory(prefix="ref_layout_") as tmp:
        wav = os.path.join(tmp, "src.wav")
        pdm = os.path.join(tmp, "src.pdm")
        stock = os.path.join(tmp, "stock.raw")
        ref = os.path.join(tmp, "ref.raw")
        # Asymmetric channels: a layout bug swaps 1 kHz and 3 kHz.

        A24.write_stereo_wav_bits(
            wav, sample_rate=48000, frames=96000,
            left_hz=1000.0, right_hz=3000.0, amplitude=0.5, bits=24,
        )
        rc, err = run([WAV2PDM, "-f3072000", "-d64", "-o2"], stdin=wav, stdout=pdm)
        assert rc == 0, err
        rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c2", "-b24"], stdin=pdm, stdout=stock)
        assert rc == 0, err
        with open(pdm, "rb") as fin, open(ref, "wb") as fout:
            p = subprocess.run(
                [REF_DECIMATOR, "-f3072000", "-d64", "-c2", "-b24",
                 "-t16", "-z8"],
                stdin=fin, stdout=fout, stderr=subprocess.PIPE, timeout=60,
            )
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")

        def dominant_freq(samples):
            seg = samples[24000:] - samples[24000:].mean()
            mid = seg[len(seg) // 4: 3 * len(seg) // 4]
            crossings = sum(1 for i in range(1, len(mid)) if mid[i - 1] <= 0 < mid[i])
            return crossings / (len(mid) / 48000.0)


        left, right = A24.read_channels(ref, 24)
        assert abs(dominant_freq(left) - 1000.0) < 60, dominant_freq(left)
        assert abs(dominant_freq(right) - 3000.0) < 60, dominant_freq(right)
        print("OK ref decimator decodes the same layout as pdm2pcm")


def test_24bit_advantage_tracks_d_and_filter():
    """24-bit gain follows one model across every decoder and rate.

        advantage = sqrt(err24^2 + 0.2887^2) / err24

    where err is the non-sinusoidal residual in 16-bit LSB. Asserting the
    model rather than one number keeps this a test of the conclusion: it
    holds whether the measured benefit is 0.9 dB or 3.8 dB.
    """
    if not os.path.isfile(REF_DECIMATOR) or not os.access(REF_DECIMATOR, os.X_OK):
        print("SKIP 24-bit advantage: ref decimator not built")
        return
    with tempfile.TemporaryDirectory(prefix="advantage_") as tmp:
        for d in (64, 128):
            rate = 3072000 * d // 64
            wav = os.path.join(tmp, f"w{d}.wav")
            pdm = os.path.join(tmp, f"p{d}.pdm")
            A24.write_stereo_wav_bits(
                wav, sample_rate=48000, frames=96000,
                left_hz=1000.0, right_hz=3000.0, amplitude=0.5, bits=24,
            )
            rc, err = run(
                [WAV2PDM, "-f", str(rate), "-d", str(d), "-o2"],
                stdin=wav, stdout=pdm,
            )
            assert rc == 0, err

            for name in ("stock", "kaiser"):
                errs = {}
                for bits in (16, 24):
                    out = os.path.join(tmp, f"{name}{bits}.raw")
                    if name == "stock":
                        rc, err = run(
                            [PDM2PCM, "-f", str(rate), "-d", str(d),
                             "-c2", "-b", str(bits)],
                            stdin=pdm, stdout=out,
                        )
                        assert rc == 0, err
                    else:
                        with open(pdm, "rb") as fin, open(out, "wb") as fout:
                            p = subprocess.run(
                                [REF_DECIMATOR, "-f", str(rate), "-d", str(d),
                                 "-c2", "-b", str(bits), "-t16", "-z8"],
                                stdin=fin, stdout=fout,
                                stderr=subprocess.PIPE, timeout=60,
                            )
                        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
                    left, _ = A24.read_channels(out, bits)
                    errs[bits] = non_sinusoidal_residual(
                        left, bits, 1000.0, 48000
                    )["rms_16lsb"]

                adv = errs[16] / errs[24]
                pred = math.sqrt(errs[24] ** 2 + 0.2887 ** 2) / errs[24]
                assert abs(adv - pred) / pred < 0.05, (
                    f"{name} d={d}: measured {adv:.3f}x but model says {pred:.3f}x"
                )
                # The benefit only shows up once the chain floor drops to the
                # 16-bit grid. d=64 stays above it, d=128 reaches it.
                if d == 128:
                    assert adv > 1.05, f"{name} d=128 should benefit: {adv}"
                else:
                    assert adv < 1.05, f"{name} d=64 should not benefit: {adv}"
                print(f"      {name:>7s} d={d:>3d}: 16b={errs[16]:.3f} "
                      f"24b={errs[24]:.3f} 24bit={adv:.3f}x")
    print("OK 24-bit advantage follows the residual model")


def test_noise_floor_window_invariance():
    """The projection floor must not depend on window length or offset.

    An earlier metric subtracted the window mean and fit only [cos, sin].
    Over a window that is not a whole number of tone periods the cos/sin
    basis has non-zero mean, so the fundamental coupled into the residual
    DC: an 8192-sample window (170.67 periods of 1 kHz) then read ~25 LSB16
    instead of ~1.4, and 8192-sample block statistics showed a phantom
    "-7/-37/-52 LSB16 cycling" that a pure sine reproduces exactly -- there
    was never a modulator DC limit cycle.  With DC inside the fit basis the
    tone is removed exactly for any window, which is what this test locks:
    the floor must agree across lengths (including non-period multiples
    like 8192) and across window offsets.

    Dropping the constant column from the basis is the mutation this test
    catches: 8192 jumps back to ~25 LSB16 and fails the < 2.0 bound.
    """
    with tempfile.TemporaryDirectory(prefix="wininv_") as tmp:
        wav = os.path.join(tmp, "src.wav")
        pdm = os.path.join(tmp, "src.pdm")
        out = os.path.join(tmp, "s24.raw")
        A24.write_stereo_wav_bits(
            wav, sample_rate=48000, frames=96000,
            left_hz=1000.0, right_hz=3000.0, amplitude=0.5, bits=24,
        )
        rc, err = run([WAV2PDM, "-f3072000", "-d64", "-o2"], stdin=wav, stdout=pdm)
        assert rc == 0, err
        rc, err = run([PDM2PCM, "-f3072000", "-d64", "-c2", "-b24"], stdin=pdm, stdout=out)
        assert rc == 0, err
        left, _ = A24.read_channels(out, 24)

        cases = [(24000, n) for n in (8192, 12288, 32768, 49152)]
        cases += [(skip, 49152) for skip in (24576, 30000, 32768)]
        floors = {
            (skip, n): non_sinusoidal_residual(
                left, 24, 1000.0, 48000, skip=skip, n=n
            )["rms_16lsb"]
            for skip, n in cases
        }
        values = list(floors.values())
        assert max(values) < 2.0, floors
        assert max(values) / min(values) < 1.15, (
            f"floor depends on window geometry: {floors}"
        )
        print(f"OK noise floor window-invariant "
              f"(7 length/offset combos: {min(values):.3f}..{max(values):.3f})")


def test_analysis_24bit_source_comparison():
    """A real 24-bit source must flow through and stay chain-limited.

    This locks in two documented facts at once: wav2pdm accepts S24 input
    and scales it correctly (the fitted 16-bit gain is ~256, not ~1), and
    the decoder chain error is far coarser than the 16-bit output grid, so
    neither output width is quantization-limited.
    """
    with tempfile.TemporaryDirectory(prefix="pdm24_artifacts_") as tmp:
        p = subprocess.run(
            [
                sys.executable,
                ANALYZE_24BIT,
                "--seconds", "2",
                # A 49152-sample window is 1024 whole periods at 1 kHz, so
                # the fundamental lands exactly on an FFT bin and THD+N
                # stays clean (a non-period window splits the tone across
                # bins and collapses THD+N to ~-21.8 dB).  The projection
                # floor itself no longer cares about window geometry now
                # that DC is inside the fit basis; 49152 is kept for the
                # FFT metrics.
                "--fft-size", "49152",
                "--skip-ms", "500",
                "--max-lag", "1200",
                "--source-bits", "24",
                "--order", "2",
                "--wav-out",
                "--keep-dir", tmp,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
        )
        assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
        out = p.stdout.decode("utf-8", "replace")
        assert "source=S24_LE" in out, out
        print("  s24 source analysis:", [line.strip() for line in out.splitlines() if "bits=16" in line][0])

        # One left-channel line per output width: bits=16 first, then bits=24.
        left_lines = [line for line in out.splitlines() if re.match(r"\s+L: THD\+N=", line)]
        assert len(left_lines) == 2, out
        gains = [float(re.search(r"gain=([\d.]+)", line).group(1)) for line in left_lines]
        floors = [float(re.search(r"noise_floor_16lsb=([\d.]+)", line).group(1)) for line in left_lines]
        assert 200.0 < gains[0] < 320.0, f"16-bit output not scaled vs 24-bit source: {gains[0]}"
        assert 0.9 < gains[1] < 1.1, f"24-bit output gain off: {gains[1]}"
        # The projection metric must land near the true chain floor.  Under
        # the old integer-lag metric this read 139.7 LSB16, of which ~136
        # was fractional-delay phase residue, so it was ~100x too high.
        assert all(0.5 < value < 5.0 for value in floors), (
            f"chain noise floor out of expected range: {floors}"
        )
        # Both widths report in 16-bit LSB, so at d=64 the two floors are
        # nearly equal: 16-bit is nowhere near being the limit.
        assert abs(floors[0] / floors[1] - 1.0) < 0.05, (
            f"16/24 floors should match at d=64: {floors}"
        )

        deltas = [float(value) for value in re.findall(r"delta_rms_16lsb=([\d.]+)", out)]
        assert len(deltas) == 2, out
        assert all(0.2 < value < 0.5 for value in deltas), f"unexpected 16/24 delta: {deltas}"

        # --wav-out must leave playable WAVs next to every raw payload.
        expected = {
            "stereo_1k3k_s24.wav": 3,
            "order2-src24.pdm": None,
            "order2-src24-16.raw": None,
            "order2-src24-24.raw": None,
            "order2-src24-16.wav": 2,
            "order2-src24-24.wav": 3,
        }
        for name, width in expected.items():
            path = os.path.join(tmp, name)
            assert os.path.isfile(path), f"missing artifact {name}: {sorted(os.listdir(tmp))}"
            if name.endswith(".wav"):
                with wave.open(path, "rb") as handle:
                    assert handle.getsampwidth() == width, f"{name} sample width"
                    assert handle.getnchannels() == 2, f"{name} channels"
                    assert handle.getframerate() == 48000, f"{name} sample rate"
        # The WAV wrapper must contain exactly the raw payload.
        for bits in (16, 24):
            raw = os.path.join(tmp, f"order2-src24-{bits}.raw")
            wav = os.path.join(tmp, f"order2-src24-{bits}.wav")
            assert os.path.getsize(wav) == os.path.getsize(raw) + 44, f"{bits}-bit wav header"
            with open(raw, "rb") as source, open(wav, "rb") as target:
                assert source.read() == target.read()[44:], f"{bits}-bit wav payload"
    print("OK 24-bit source comparison is chain-limited and leaves playable WAVs")


def test_doc_numbers_match_measurements():
    """Every figure in the three markdown files must re-measure correctly.

    Three wrong conclusions in this project came out of a broken measurement
    rather than broken code, and twice the article carried a number that no
    longer matched the code.  check_doc_numbers.py re-derives each figure from
    the binaries and compares it against the literal text, so a doc edited by
    hand fails here instead of misleading a reader.

    Skipped when the docs are not checked out alongside the source, which is
    the normal case for someone who only wants the tool.
    """
    if not os.path.isfile(CHECK_DOC_NUMBERS):
        print("SKIP doc numbers: check_doc_numbers.py not present")
        return
    docs = None
    for cand in (
        os.path.join(HERE, "..", "..", "pdm2pcm", "docs", "eli5"),
        os.path.join(HERE, "..", "docs", "eli5"),
        os.path.join(HERE, "docs", "eli5"),
    ):
        cand = os.path.normpath(cand)
        if os.path.isfile(os.path.join(cand, "PDM-24bit输出实测对比.md")):
            docs = cand
            break
    if docs is None:
        print("SKIP doc numbers: docs/eli5 not checked out alongside the source")
        return
    p = subprocess.run(
        [sys.executable, CHECK_DOC_NUMBERS, "--docs", docs],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=1800,
    )
    out = p.stdout.decode("utf-8", "replace")
    assert p.returncode == 0, out[-3000:]
    assert "ALL DOC NUMBERS MATCH MEASUREMENTS" in out, out[-500:]
    checks = out.count("  OK   ")
    print(f"OK doc numbers: {checks} figures re-measured and matched")


def test_reproduction_script_self_test():
    assert os.path.isfile(ANALYZE_PDM), "analyze_pdm.py is missing"
    p = subprocess.run(
        [sys.executable, ANALYZE_PDM, "--self-test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")
    assert b"self-test passed" in p.stdout.lower(), p.stdout.decode("utf-8", "replace")
    print("OK reproduction script self-test")


def main():
    assert os.path.isfile(WAV2PDM) and os.access(WAV2PDM, os.X_OK), "build wav2pdm first"
    assert os.path.isfile(PDM2PCM) and os.access(PDM2PCM, os.X_OK), "build pdm2pcm first"
    test_cli_rejects()
    test_wav_odd_chunk_padding()
    test_wav2pdm_accepts_s24_source()
    test_pdm2pcm_c_rejects()
    test_pdm2pcm_strict_arguments()
    test_pdm2pcm_24bit_output()
    test_pdm2pcm_24bit_fullscale_saturates()
    test_pdm2pcm_rejects_invalid_buffer_geometry()
    test_pdm2pcm_rejects_unsupported_pcm_rate()
    test_pdm2pcm_rejects_partial_block()
    test_order2_fullscale_ubsan()
    test_24bit_decode_fullscale_ubsan()
    test_pdm2pcm_cic_n_args()
    test_pdm2pcm_cic9_fullscale_bounds()
    test_pdm2pcm_cic9_advantage()
    test_order_option_changes_pdm_stream()
    test_stereo_silence_bitpattern()
    test_pack_bitpattern()
    for order in (1, 2):
        roundtrip(1024000, 128, order=order)
        roundtrip(512000, 64, order=order)
        roundtrip_stereo(1024000, 128, order=order)
        roundtrip_stereo(512000, 64, order=order)
    test_analysis_script_coherent_measurement_defaults()
    test_analysis_lag_uses_overlap_correlation()
    test_analysis_script_timeout_and_input_validation()
    test_non_sinusoidal_residual_metric()
    test_analysis_24bit_script_self_test()
    test_analysis_24bit_source_comparison()
    test_ref_decimator_matches_stock_layout()
    test_24bit_advantage_tracks_d_and_filter()
    test_noise_floor_window_invariance()
    test_doc_numbers_match_measurements()
    test_reproduction_script_self_test()
    print("ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
