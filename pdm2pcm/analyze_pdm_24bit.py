#!/usr/bin/env python3
# Copyright (c) 2026 piaozhiye <piaozhiye@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Compare the native 16-bit and 24-bit pdm2pcm output paths.

The same packed PDM stream is decoded twice, once with -b 16 and once
with -b 24.  This keeps the Sigma-Delta input identical and measures the
output quantization difference rather than re-encoding differences.

With --source-bits 24 the fixture WAV itself carries 24-bit samples, so
wav2pdm encodes a 24-bit signal and both PCM outputs are compared against
the true 24-bit reference instead of a 16-bit one.
"""
import argparse
import math
import os
import struct
import sys
import tempfile
import wave

import numpy as np

import analyze_pdm as base

HERE = os.path.dirname(os.path.abspath(__file__))
WAV2PDM = os.path.join(HERE, "wav2pdm")
PDM2PCM = os.path.join(HERE, "pdm2pcm")


def decode_s24le(data):
    """Decode packed signed 24-bit little-endian bytes into int64 samples."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size % 3:
        raise ValueError("24-bit PCM payload is not 3-byte aligned")
    raw = raw.reshape(-1, 3).astype(np.int64)
    values = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
    return np.where(values & 0x800000, values - 0x1000000, values)


def read_s24le(path):
    """Read packed signed 24-bit little-endian samples as int64."""
    with open(path, "rb") as stream:
        return decode_s24le(stream.read())


def read_channels(path, bits):
    if bits == 16:
        return base.read_raw_channels(path)
    samples = read_s24le(path)
    if samples.size % 2:
        raise ValueError("24-bit PCM payload is not frame aligned")
    return samples[0::2].astype(np.float64), samples[1::2].astype(np.float64)


def pack_s24le(values):
    """Pack signed values into packed signed 24-bit little-endian bytes."""
    unsigned = np.asarray(values, dtype=np.int64) & 0xFFFFFF
    raw = np.empty((unsigned.size, 3), dtype=np.uint8)
    raw[:, 0] = (unsigned >> 0) & 0xFF
    raw[:, 1] = (unsigned >> 8) & 0xFF
    raw[:, 2] = (unsigned >> 16) & 0xFF
    return raw.tobytes()


def write_stereo_wav_bits(path, sample_rate, frames, left_hz, right_hz, amplitude, bits):
    """Write a deterministic stereo sine WAV at 16-bit or packed 24-bit."""
    full_scale = 32767.0 if bits == 16 else 8388607.0
    left = [
        int(amplitude * full_scale * math.sin(2.0 * math.pi * left_hz * n / sample_rate))
        for n in range(frames)
    ]
    right = [
        int(amplitude * full_scale * math.sin(2.0 * math.pi * right_hz * n / sample_rate))
        for n in range(frames)
    ]
    with wave.open(path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(bits // 8)
        wav.setframerate(sample_rate)
        if bits == 16:
            payload = bytearray()
            for a, b in zip(left, right):
                payload.extend(struct.pack("<hh", a, b))
            wav.writeframes(bytes(payload))
        else:
            flat = np.empty(frames * 2, dtype=np.int64)
            flat[0::2] = left
            flat[1::2] = right
            wav.writeframes(pack_s24le(flat))


def read_wav_channels_bits(path, bits):
    """Read a 16-bit or packed 24-bit stereo WAV into float64 channels."""
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 2 or wav.getsampwidth() != bits // 8:
            raise ValueError(f"analysis input must be {bits}-bit stereo WAV")
        sample_rate = wav.getframerate()
        payload = wav.readframes(wav.getnframes())
    if bits == 16:
        samples = np.frombuffer(payload, dtype="<i2")
    else:
        samples = decode_s24le(payload)
    if samples.size % 2:
        raise ValueError("WAV payload is not frame aligned")
    samples = samples.astype(np.float64)
    return sample_rate, samples[0::2], samples[1::2]


def low_byte_fraction(samples):
    values = np.asarray(samples, dtype=np.int64)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values & 0xFF) / values.size)


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    return math.sqrt(float(np.mean(values * values)))


def write_pcm_wav(raw_path, bits, sample_rate):
    """Wrap a decoded raw PCM payload in a WAV header so it can be played."""
    with open(raw_path, "rb") as stream:
        payload = stream.read()
    wav_path = os.path.splitext(raw_path)[0] + ".wav"
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(bits // 8)
        wav.setframerate(sample_rate)
        wav.writeframes(payload)
    return wav_path


def non_sinusoidal_rms_16lsb(decoded, sample_rate, args, out_bits, fundamental):
    """Decode-chain noise floor in 16-bit LSB, with the fundamental removed.

    This is the number that shows whether an output is limited by its own
    output quantization grid: a 16-bit output cannot beat roughly
    1/sqrt(12) ~= 0.289 LSB16 against a higher-resolution reference.

    The fundamental is projected out with a least-squares cos/sin fit rather
    than aligned against the source by integer lag.  An integer-lag alignment
    leaves the filter's fractional group delay (0.09 of a sample here) in
    place, and at 1 kHz that alone contributed ~136 LSB16 of phase residue
    -- 500x the 16-bit grid, which hid the bit-depth effect completely.

    out_bits is the width of `decoded`: int16 samples are already in 16-bit
    LSB, int32 samples are 256x finer and get divided down.
    """
    floor = base.project_out_fundamental(
        decoded, fundamental, sample_rate, args.skip_ms, args.fft_size
    )
    value = floor["residual_rms"]
    if out_bits == 24:
        value /= 256.0
    return value


def channel_metrics(reference, decoded, sample_rate, fundamental, args, out_bits):
    fft = base.fft_metrics(
        decoded,
        sample_rate,
        fundamental,
        args.fft_size,
        args.skip_ms,
    )
    residual = base.residual_snr_db(
        reference,
        decoded,
        sample_rate,
        args.skip_ms,
        args.fft_size,
        args.max_lag,
    )
    return {
        "thdn_db": fft["thdn_db"],
        "harmonic_db": fft["harmonic_db"],
        "residual_snr_db": residual["snr_db"],
        "lag": residual["lag"],
        "correlation": residual["correlation"],
        "gain": residual["gain"],
        "noise_floor_16lsb": non_sinusoidal_rms_16lsb(
            decoded, sample_rate, args, out_bits, fundamental
        ),
    }


def validate(args):
    numeric = (args.seconds, args.left_hz, args.right_hz, args.amplitude)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("floating-point options must be finite")
    if args.sample_rate <= 0 or args.pdm_rate <= 0 or args.fft_size < 16:
        raise ValueError("sample rates must be positive and FFT size must be >= 16")
    if args.seconds <= 0 or args.skip_ms < 0 or args.max_lag < 0:
        raise ValueError("seconds must be positive; skip_ms/max_lag non-negative")
    if args.amplitude <= 0 or args.amplitude > 1:
        raise ValueError("amplitude must be in (0, 1]")
    if args.decimation not in (64, 128):
        raise ValueError("decimation must be 64 or 128")
    if args.source_bits not in (16, 24):
        raise ValueError("source bits must be 16 or 24")
    if args.sample_rate != args.pdm_rate // args.decimation:
        raise ValueError("sample rate must equal PDM rate / decimation")
    if args.pdm_rate % (1000 * args.decimation):
        raise ValueError("PDM rate must be divisible by 1000 * decimation")
    required = args.skip_ms * args.sample_rate / 1000.0 + args.fft_size + 1
    if args.seconds * args.sample_rate < required:
        raise ValueError("seconds does not leave enough samples for FFT")
    frames_per_block = args.pdm_rate // (1000 * args.decimation)
    if int(args.seconds * args.sample_rate) % frames_per_block:
        raise ValueError("seconds does not produce whole 1 ms decoder blocks")


def analyze(args):
    validate(args)
    for binary in (WAV2PDM, PDM2PCM):
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(f"build the project first; missing executable: {binary}")

    temporary = None
    if args.keep_dir:
        os.makedirs(args.keep_dir, exist_ok=True)
        work_dir = args.keep_dir
    else:
        temporary = tempfile.TemporaryDirectory(prefix="pdm24-analysis-")
        work_dir = temporary.name

    try:
        frames = int(args.seconds * args.sample_rate)
        source_bits = args.source_bits
        wav_path = os.path.join(work_dir, f"stereo_1k3k_s{source_bits}.wav")
        write_stereo_wav_bits(
            wav_path,
            sample_rate=args.sample_rate,
            frames=frames,
            left_hz=args.left_hz,
            right_hz=args.right_hz,
            amplitude=args.amplitude,
            bits=source_bits,
        )
        sample_rate, reference_left, reference_right = read_wav_channels_bits(
            wav_path, source_bits
        )
        print(
            f"fixture: rate={sample_rate} frames={frames} source=S{source_bits}_LE "
            f"L={args.left_hz}Hz R={args.right_hz}Hz amplitude={args.amplitude}"
        )
        print("decode: same PDM bytes, -b 16 vs -b 24; no re-encoding between paths")
        print("metrics: THD+N=non-DC residual/fundamental; residual-SNR=aligned least-squares fit")
        print("noise_floor_16lsb=decode-chain noise floor with the fundamental projected out, in 16-bit LSB")
        print("  (projecting the fundamental, not integer-lag alignment: a 0.09-sample fractional")
        print("   group delay would otherwise add ~136 LSB16 of pure phase residue)")
        print("low_byte_fraction=nonzero low 8 bits; 24-bit values are scaled by 1/256 vs 16-bit path")

        orders = args.order if args.order is not None else [1, 2]
        for order in orders:
            pdm_path = os.path.join(work_dir, f"order{order}-src{source_bits}.pdm")
            base.run_tool(
                [WAV2PDM, "-f", str(args.pdm_rate), "-d", str(args.decimation), "-o", str(order)],
                wav_path,
                pdm_path,
            )
            print(f"\norder={order} PDM_bytes={os.path.getsize(pdm_path)}")
            results = {}
            for bits in (16, 24):
                raw_path = os.path.join(work_dir, f"order{order}-src{source_bits}-{bits}.raw")
                command = [
                    PDM2PCM,
                    "-f", str(args.pdm_rate),
                    "-d", str(args.decimation),
                    "-c", "2",
                    "-b", str(bits),
                ]
                base.run_tool(command, pdm_path, raw_path)
                if args.wav_out:
                    print(f"  wav={write_pcm_wav(raw_path, bits, sample_rate)}")
                decoded_left, decoded_right = read_channels(raw_path, bits)
                results[bits] = (decoded_left, decoded_right)
                print(f"  bits={bits} output_bytes={os.path.getsize(raw_path)}")
                for label, reference, decoded, fundamental in (
                    ("L", reference_left, decoded_left, args.left_hz),
                    ("R", reference_right, decoded_right, args.right_hz),
                ):
                    metrics = channel_metrics(
                        reference, decoded, sample_rate, fundamental, args, bits
                    )
                    print(
                        f"    {label}: THD+N={metrics['thdn_db']:.2f}dB "
                        f"harmonic_only={metrics['harmonic_db']:.2f}dB "
                        f"residual_SNR={metrics['residual_snr_db']:.2f}dB "
                        f"noise_floor_16lsb={metrics['noise_floor_16lsb']:.4f} "
                        f"lag={metrics['lag']} corr={metrics['correlation']:.5f} "
                        f"gain={metrics['gain']:.6f}"
                    )
                if bits == 24:
                    print(
                        f"    low_byte_fraction L={low_byte_fraction(decoded_left):.4f} "
                        f"R={low_byte_fraction(decoded_right):.4f}"
                    )

            left16, right16 = results[16]
            left24, right24 = results[24]
            for label, decoded16, decoded24 in (("L", left16, left24), ("R", right16, right24)):
                delta = decoded24 / 256.0 - decoded16
                print(
                    f"    24_vs_16 {label}: delta_rms_16lsb={rms(delta):.6f} "
                    f"delta_peak_16lsb={np.max(np.abs(delta)):.3f}"
                )
    finally:
        if temporary is not None:
            temporary.cleanup()


def self_test():
    with tempfile.TemporaryDirectory(prefix="pdm24-selftest-") as tmp:
        path = os.path.join(tmp, "sample.s24")
        values = np.array([0, 1, -1, 0x7FFFFF, -0x800000], dtype=np.int64)
        with open(path, "wb") as stream:
            for value in values:
                unsigned = value & 0xFFFFFF
                stream.write(bytes((unsigned & 0xFF, (unsigned >> 8) & 0xFF, (unsigned >> 16) & 0xFF)))
        if not np.array_equal(read_s24le(path), values):
            raise AssertionError("S24_LE self-test decode failed")
        if not np.array_equal(decode_s24le(pack_s24le(values)), values):
            raise AssertionError("pack_s24le self-test failed")

        wav_path = os.path.join(tmp, "fixture_s24.wav")
        write_stereo_wav_bits(
            wav_path,
            sample_rate=48000,
            frames=64,
            left_hz=1000.0,
            right_hz=3000.0,
            amplitude=0.5,
            bits=24,
        )
        rate, left, right = read_wav_channels_bits(wav_path, 24)
        if rate != 48000 or left.size != 64 or right.size != 64:
            raise AssertionError("24-bit fixture WAV self-test shape failed")
        if int(np.max(np.abs(left))) <= 1000:
            raise AssertionError("24-bit fixture WAV self-test amplitude too small")
    base.self_test()
    print("24-bit self-test passed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run format/FFT self-tests")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--pdm-rate", type=int, default=3072000)
    parser.add_argument("--decimation", type=int, choices=(64, 128), default=64)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--skip-ms", type=int, default=500)
    parser.add_argument("--fft-size", type=int, default=49152)
    parser.add_argument("--max-lag", type=int, default=2000)
    parser.add_argument("--left-hz", type=float, default=1000.0)
    parser.add_argument("--right-hz", type=float, default=3000.0)
    parser.add_argument("--amplitude", type=float, default=0.5)
    parser.add_argument(
        "--source-bits",
        type=int,
        choices=(16, 24),
        default=16,
        help="bit depth of the generated fixture WAV (16 or 24)",
    )
    parser.add_argument("--order", type=int, choices=(1, 2), action="append")
    parser.add_argument("--keep-dir", help="keep generated WAV/PDM/raw files in this directory")
    parser.add_argument(
        "--wav-out",
        action="store_true",
        help="also wrap each decoded raw PCM payload in a playable WAV file",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    analyze(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
