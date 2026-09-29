# PDM decoding on PC

This directory contains the existing PDM acquisition helpers plus a
closed-loop `wav2pdm` → `pdm2pcm` path for 16-bit and 24-bit mono/stereo
WAV files.

`pdm2pcm` decodes packed PDM bytes to signed 16-bit PCM by default. Use
`-b 24` to request native signed 24-bit output packed as little-endian
three-byte samples (`S24_LE`).

## Author and license

Author: piaozhiye <piaozhiye@gmail.com>
Copyright (c) 2026 piaozhiye

This project is licensed under the Apache License, Version 2.0. See the
repository `LICENSE` and `NOTICE` files for details.

## Build and tests

```sh
make
python3 test_wav2pdm.py
```

The test suite covers CLI validation, mono/stereo bit packing, d=64 and
d=128 round-trips, first- and second-order encoding, a UBSan full-scale
second-order run, 24-bit PCM output, 24-bit WAV input, and invalid
`pdm2pcm` block geometry.

## WAV → PDM → PCM

`wav2pdm` reads S16 or S24 little-endian PCM WAV from stdin and detects
the sample width from the `fmt ` chunk. The WAV sample rate must equal
`f/d`; the PDM rate is `f` and the decimation factor is 64 or 128.

```sh
# First order (default)
./wav2pdm -f3072000 -d64 < stereo.wav > stereo.pdm

# Second order
./wav2pdm -f3072000 -d64 -o 2 < stereo.wav > stereo.pdm
```

A 24-bit source is encoded at its own full scale: the modulator
thresholds follow the input width (`32768`/`128` for 16-bit,
`8388608`/`32768` for 24-bit), so 24-bit samples are not driven as if
they were 16-bit.

The `-o` value is applied to every input sample and to each stereo channel:

- `1`: first-order Sigma-Delta;
- `2`: second-order Sigma-Delta with explicitly bounded wide states.

The encoder accepts mono or stereo input. For stereo input it writes
`L-byte, R-byte, L-byte, R-byte, ...`; mono input writes one packed byte
stream. The decoder must be told the channel count because the packed PDM
stream has no channel header:

```sh
./pdm2pcm -f3072000 -d64 -c2 < stereo.pdm > stereo.raw
```

Mono is the default (`-c1` or omitted):

```sh
./pdm2pcm -f3072000 -d64 -c1 < mono.pdm > mono.raw
```

### PCM output bit depth

The default is signed 16-bit little-endian PCM:

```sh
./pdm2pcm -f3072000 -d64 -c1 < mono.pdm > mono.s16.raw
aplay -f S16_LE -c 1 -r 48000 mono.s16.raw
```

Use `-b 24` for native signed 24-bit output packed as three-byte
little-endian samples:

```sh
./pdm2pcm -f3072000 -d64 -c1 -b 24 < mono.pdm > mono.s24.raw
aplay -f S24_LE -c 1 -r 48000 mono.s24.raw
```

`-b 16` and `-b 24` are the supported values. The 24-bit path keeps the
filter's internal headroom and applies a 24-bit scale before saturation;
it is not a 16-bit sample shifted by eight bits.

### CIC order (-n 3 | 9)

`-n` selects the number of CIC integrator stages. `-n 3` (the default) is
the stock three-stage kernel and is bit-identical to previous releases.
`-n 9` uses a nine-stage kernel: the stopband improves by ~107 dB and the
measured decode noise floor drops from 0.254 to 0.071 LSB16 at d=128, which
pushes it below the 16-bit quantization grid -- so `-n 9` only pays off
combined with `-b 24` (measured 24-bit benefit at d=128: 4.35x / 12.8 dB in
the chain residual; at d=64: 1.27x / 2.1 dB). Note the bare 9-stage kernel
has ~-24 dB passband droop at 20 kHz; acoustic measurements remain
microphone-noise limited either way.

Internal widths follow Hogenauer's W2 = W1 + N*log2(R*D): the stage
registers and div_const are 64-bit, and the 24-bit quantizer divides before
multiplying when div_const is large, so no intermediate wraps for any
supported order/rate combination (verified by the UBSan full-scale sweep).

`pdm2pcm` processes complete 1 ms blocks, so the PDM rate must satisfy
`f % (1000 * d) == 0`, and the resulting PCM rate must fit the filter API's
`uint16_t` sample-rate field. Invalid rates are rejected instead of producing
a zero-length output or looping forever. A trailing partial block is reported
as an error after any complete preceding blocks have been emitted.

## Reproducible measurements

`analyze_pdm.py` generates a deterministic 48 kHz stereo fixture with
1 kHz on the left channel and 3 kHz on the right channel, then runs both
Sigma-Delta orders and prints the measurements:

```sh
# Verify the FFT helper without running the C tools
python3 analyze_pdm.py --self-test

# Run the complete reproduction
python3 analyze_pdm.py --seconds 2
```

The script requires NumPy. Its output uses these definitions:

- `harmonic_only`: power in bins around harmonics 2..10 that fall below
  Nyquist divided by the fundamental power;
- `THD+N`: all non-DC FFT power except the fundamental divided by the
  fundamental power;
- `residual_SNR`: reference-to-decoded residual after integer-delay
  alignment and a least-squares gain fit.

`residual_SNR` is a reproducible comparison metric for this toolchain; it
is not a microphone datasheet SNR. Use the script output rather than
copying an old hand-written number into documentation. The default
49152-point, 48 kHz FFT makes the 1 kHz and 3 kHz tones coherent; its bin
width is 0.9765625 Hz.

Generated files are temporary by default. Pass `--keep-dir DIR` to retain
the WAV, PDM, and raw PCM files for inspection.

## Existing logic-analyzer examples

- Acquire PDM data using a Logic Analyzer and export the data in ASCII
  format. End result is a text file containing single rows of zeroes and
  ones. If using Salae logic analyzer, for example, you can use the
  "Parallel Decoder". See `bellazio.txt.bz2` for a PDM dump file example.
- Convert PDM data in the format suitable for OpenPDM library, i.e. packed
  PDM bits, using `txt2bin.c`:

  ```sh
  bzcat bellazio.txt.bz2 | ./txt2bin > bellazio.dat
  ```

  The bundled capture ends with a partial 128-byte block. The decoder
  intentionally reports that as an error, so trim it before decoding:

  ```sh
  bytes=$(wc -c < bellazio.dat)
  head -c "$((bytes / 128 * 128))" bellazio.dat | \
    ./pdm2pcm -f1024000 -d128 | aplay -fS16_LE -c1 -r8000
  ```
- Follow the instructions in `pdmgrabber` for the Cypress FX2LP setup.
  Use `fx2grabber` to capture PDM and `packdata` to extract a microphone
  channel before decoding.
