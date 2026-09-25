#!/usr/bin/env python3
# Copyright (c) 2026 piaozhiye <piaozhiye@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Reproducible WAV -> PDM -> PCM measurements.

The script generates a deterministic stereo test WAV, runs the local
wav2pdm/pdm2pcm binaries for both Sigma-Delta orders, and reports metrics
with their definitions.  NumPy is used for the FFT and least-squares fit.
"""

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("analyze_pdm.py requires NumPy: python3 -m pip install numpy") from exc


HERE = os.path.dirname(os.path.abspath(__file__))
WAV2PDM = os.path.join(HERE, "wav2pdm")
PDM2PCM = os.path.join(HERE, "pdm2pcm")


def write_stereo_wav(path, sample_rate, frames, left_hz, right_hz, amplitude):
    """Write a deterministic 16-bit stereo sine WAV."""
    with wave.open(path, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        payload = bytearray()
        for n in range(frames):
            left = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * left_hz * n / sample_rate))
            right = int(amplitude * 32767.0 * math.sin(2.0 * math.pi * right_hz * n / sample_rate))
            payload.extend(struct.pack("<hh", left, right))
        wav.writeframes(payload)


def read_wav_channels(path):
    """Read a 16-bit stereo WAV and return int16-valued float64 channels."""
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 2 or wav.getsampwidth() != 2:
            raise ValueError("analysis input must be 16-bit stereo WAV")
        sample_rate = wav.getframerate()
        payload = wav.readframes(wav.getnframes())
    samples = np.frombuffer(payload, dtype="<i2")
    if samples.size % 2:
        raise ValueError("WAV payload is not frame aligned")
    samples = samples.astype(np.float64)
    return sample_rate, samples[0::2], samples[1::2]


def read_raw_channels(path):
    """Read interleaved signed 16-bit little-endian PCM."""
    samples = np.fromfile(path, dtype="<i2").astype(np.float64)
    if samples.size % 2:
        raise ValueError("decoded PCM payload is not frame aligned")
    return samples[0::2], samples[1::2]


def db20(ratio):
    if ratio <= 0.0 or not math.isfinite(ratio):
        return float("-inf")
    return 20.0 * math.log10(ratio)


def band_power(power, center, half_width=1):
    start = max(1, center - half_width)
    end = min(len(power), center + half_width + 1)
    if start >= end:
        return 0.0
    return float(np.sum(power[start:end]))


def fft_metrics(signal, sample_rate, fundamental_hz, fft_size, skip_ms, harmonics=10):
    """Return clearly defined harmonic, THD+N, and band metrics.

    THD+N is calculated as all non-DC FFT power except the fundamental
    divided by fundamental power.  harmonic_db is a separate diagnostic
    that sums only bins around harmonics 2..harmonics.
    """
    skip = int(round(sample_rate * skip_ms / 1000.0))
    if len(signal) < skip + fft_size:
        raise ValueError(
            f"need {skip + fft_size} samples for FFT, got {len(signal)}"
        )
    segment = signal[skip:skip + fft_size]
    windowed = (segment - np.mean(segment)) * np.hanning(fft_size)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    bin_hz = sample_rate / fft_size
    fundamental_bin = int(round(fundamental_hz / bin_hz))
    if fundamental_bin <= 0 or fundamental_bin >= len(power) - 1:
        raise ValueError("fundamental is outside the FFT band")

    fundamental_power = band_power(power, fundamental_bin)
    harmonic_power = 0.0
    harmonic_count = 0
    for harmonic in range(2, harmonics + 1):
        center = int(round(fundamental_hz * harmonic / bin_hz))
        if center + 1 >= len(power):
            break
        harmonic_power += band_power(power, center)
        harmonic_count += 1
    total_non_dc_power = float(np.sum(power[1:]))
    residual_power = max(0.0, total_non_dc_power - fundamental_power)
    return {
        "fundamental_bin_hz": fundamental_bin * bin_hz,
        "fundamental_power": fundamental_power,
        "harmonic_db": db20(math.sqrt(harmonic_power / fundamental_power)),
        "thdn_db": db20(math.sqrt(residual_power / fundamental_power)),
        "harmonic_count": harmonic_count,
        "bin_hz": bin_hz,
    }


def best_positive_lag(reference, decoded, max_lag):
    """Find the decoded delay that maximizes overlap-normalized correlation."""
    reference = np.asarray(reference[: min(len(reference), len(decoded))], dtype=np.float64)
    decoded = np.asarray(decoded[: len(reference)], dtype=np.float64)
    size = len(reference)
    max_lag = min(max_lag, size - 2)
    if max_lag < 0:
        raise ValueError("not enough samples for lag search")

    prefix_reference = np.concatenate(([0.0], np.cumsum(reference)))
    prefix_decoded = np.concatenate(([0.0], np.cumsum(decoded)))
    prefix_reference_sq = np.concatenate(([0.0], np.cumsum(reference * reference)))
    prefix_decoded_sq = np.concatenate(([0.0], np.cumsum(decoded * decoded)))

    fft_len = 1 << (2 * size - 1).bit_length()
    cross_correlation = np.fft.irfft(
        np.fft.rfft(decoded, fft_len) * np.conj(np.fft.rfft(reference, fft_len)),
        fft_len,
    )
    best_lag = 0
    best_correlation = -2.0
    for lag in range(max_lag + 1):
        overlap = size - lag
        sum_reference = prefix_reference[overlap]
        sum_decoded = prefix_decoded[size] - prefix_decoded[lag]
        sum_reference_sq = prefix_reference_sq[overlap]
        sum_decoded_sq = prefix_decoded_sq[size] - prefix_decoded_sq[lag]
        covariance = (
            cross_correlation[lag]
            - sum_reference * sum_decoded / overlap
        )
        variance_reference = sum_reference_sq - sum_reference * sum_reference / overlap
        variance_decoded = sum_decoded_sq - sum_decoded * sum_decoded / overlap
        denominator = math.sqrt(max(0.0, variance_reference * variance_decoded))
        correlation = covariance / denominator if denominator > 0.0 else -2.0
        if correlation > best_correlation:
            best_lag = lag
            best_correlation = correlation
    return best_lag, best_correlation


def residual_snr_db(reference, decoded, sample_rate, skip_ms, fft_size, max_lag):
    """Measure residual SNR after integer-delay alignment and gain fit."""
    skip = int(round(sample_rate * skip_ms / 1000.0))
    size = min(fft_size, len(reference) - skip, len(decoded) - skip)
    if size <= max_lag + 2:
        raise ValueError("not enough samples for residual SNR")
    reference = reference[skip:skip + size]
    decoded = decoded[skip:skip + size]
    lag, correlation = best_positive_lag(reference, decoded, max_lag)
    aligned_reference = reference[:size - lag]
    aligned_decoded = decoded[lag:]
    denominator = float(np.dot(aligned_decoded, aligned_decoded))
    if denominator == 0.0:
        raise ValueError("decoded channel has zero energy")
    gain = float(np.dot(aligned_reference, aligned_decoded) / denominator)
    residual = aligned_reference - gain * aligned_decoded
    signal_rms = math.sqrt(float(np.mean(aligned_reference * aligned_reference)))
    residual_rms = math.sqrt(float(np.mean(residual * residual)))
    return {
        "lag": lag,
        "correlation": correlation,
        "gain": gain,
        "snr_db": db20(signal_rms / residual_rms),
    }


def run_tool(command, stdin_path, stdout_path, timeout=60):
    with open(stdin_path, "rb") as source, open(stdout_path, "wb") as target:
        try:
            result = subprocess.run(
                command,
                stdin=source,
                stdout=target,
                stderr=subprocess.PIPE,
                cwd=HERE,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"command timed out after {timeout}s: {' '.join(command)}"
            ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{stderr}")


def analyze(args):
    numeric_options = (args.seconds, args.left_hz, args.right_hz, args.amplitude)
    if not all(math.isfinite(value) for value in numeric_options):
        raise ValueError("floating-point options must be finite")
    for binary in (WAV2PDM, PDM2PCM):
        if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
            raise RuntimeError(f"build the project first; missing executable: {binary}")
    if args.seconds <= 0.0 or args.skip_ms < 0 or args.max_lag < 0:
        raise ValueError("seconds must be positive; skip_ms and max_lag must be non-negative")
    if args.left_hz <= 0.0 or args.right_hz <= 0.0:
        raise ValueError("fixture frequencies must be positive")
    if args.pdm_rate <= 0 or args.pdm_rate % (1000 * args.decimation) != 0:
        raise ValueError("PDM rate must be divisible by 1000 * decimation")
    required_samples = args.skip_ms * args.sample_rate / 1000.0 + args.fft_size + 1
    if args.seconds * args.sample_rate < required_samples:
        raise ValueError("seconds must leave enough samples after skip_ms for the FFT")
    if not 0.0 < args.amplitude <= 1.0:
        raise ValueError("amplitude must be in (0, 1]")
    if args.sample_rate != args.pdm_rate // args.decimation:
        raise ValueError("sample rate must equal PDM rate / decimation")
    frames = int(args.seconds * args.sample_rate)
    frames_per_block = args.pdm_rate // (1000 * args.decimation)
    if frames % frames_per_block != 0:
        raise ValueError("seconds must produce whole 1 ms decoder blocks")

    temporary = None
    if args.keep_dir:
        os.makedirs(args.keep_dir, exist_ok=True)
        work_dir = args.keep_dir
    else:
        temporary = tempfile.TemporaryDirectory(prefix="pdm-analysis-")
        work_dir = temporary.name

    try:
        wav_path = os.path.join(work_dir, "stereo_1k3k_48k.wav")
        write_stereo_wav(
            wav_path,
            args.sample_rate,
            frames,
            args.left_hz,
            args.right_hz,
            args.amplitude,
        )
        sample_rate, reference_left, reference_right = read_wav_channels(wav_path)
        print(
            f"fixture: rate={sample_rate} frames={frames} "
            f"L={args.left_hz}Hz R={args.right_hz}Hz amplitude={args.amplitude}"
        )
        print(
            f"FFT: size={args.fft_size} skip={args.skip_ms}ms "
            f"bin={args.sample_rate / args.fft_size:.6f}Hz "
            f"THD+N=non-DC residual/fundamental; residual-SNR=aligned least-squares fit"
        )

        orders = args.order if args.order is not None else [1, 2]
        for order in orders:
            pdm_path = os.path.join(work_dir, f"order{order}.pdm")
            raw_path = os.path.join(work_dir, f"order{order}.raw")
            run_tool(
                [WAV2PDM, "-f", str(args.pdm_rate), "-d", str(args.decimation), "-o", str(order)],
                wav_path,
                pdm_path,
            )
            run_tool(
                [PDM2PCM, "-f", str(args.pdm_rate), "-d", str(args.decimation), "-c", "2"],
                pdm_path,
                raw_path,
            )
            decoded_left, decoded_right = read_raw_channels(raw_path)
            print(f"\norder={order} PDM_bytes={os.path.getsize(pdm_path)}")
            for label, reference, decoded, fundamental in (
                ("L", reference_left, decoded_left, args.left_hz),
                ("R", reference_right, decoded_right, args.right_hz),
            ):
                metrics = fft_metrics(
                    decoded,
                    args.sample_rate,
                    fundamental,
                    args.fft_size,
                    args.skip_ms,
                )
                residual = residual_snr_db(
                    reference,
                    decoded,
                    args.sample_rate,
                    args.skip_ms,
                    args.fft_size,
                    args.max_lag,
                )
                print(
                    f"  {label}: fundamental_bin={metrics['fundamental_bin_hz']:.3f}Hz "
                    f"harmonic_only={metrics['harmonic_db']:.2f}dB "
                    f"THD+N={metrics['thdn_db']:.2f}dB "
                    f"residual_SNR={residual['snr_db']:.2f}dB "
                    f"lag={residual['lag']} corr={residual['correlation']:.5f} "
                    f"gain={residual['gain']:.6f}"
                )
    finally:
        if temporary is not None:
            temporary.cleanup()


def self_test():
    sample_rate = 48000
    fft_size = 49152
    signal = 0.5 * 32767.0 * np.sin(
        2.0 * np.pi * 1000.0 * np.arange(fft_size) / sample_rate
    )
    metrics = fft_metrics(signal, sample_rate, 1000.0, fft_size, 0)
    if metrics["fundamental_power"] <= 0.0 or metrics["thdn_db"] >= -80.0:
        raise AssertionError(f"unexpected self-test metrics: {metrics}")
    print("self-test passed")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run FFT self-test only")
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
    parser.add_argument("--order", type=int, choices=(1, 2), action="append")
    parser.add_argument("--keep-dir", help="keep generated WAV/PDM/raw files in this directory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.self_test:
        numeric_options = (args.seconds, args.left_hz, args.right_hz, args.amplitude)
        if not all(math.isfinite(value) for value in numeric_options):
            raise ValueError("floating-point options must be finite")
        self_test()
        return 0
    if args.fft_size < 16 or args.sample_rate <= 0 or args.pdm_rate <= 0:
        raise ValueError("sample rates must be positive and FFT size must be >= 16")
    analyze(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
