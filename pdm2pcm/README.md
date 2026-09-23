# PDM decoding on PC

Using logic analyzer to grab PDM data

- Acquire PDM data using a Logic Analyzer and export the data in ASCII format. End result is a text file containing single rows of zeroes and ones. If using Salae logic analyzer, for example, you can use the "Parallel Decoder". See `bellazio.txt.bz2` for a PDM dump file example.
- Convert PDM data in the format suitable for openpdm library, i.e.: packed PDM bits, using 'txt2bin.c' program: `bzcat bellazio.txt.bz2 | ./txt2bin > bellazio.dat`
- Listen to decoded audio passing PDM sampling frequency (-f option) and decimation factor (-d option) : `pdm2pcm -f1024000 -d128 < ./bellazio.dat | aplay -fS16_LE -c1 -r8000`
- All in a single line: `bzcat bellazio.txt.bz2 | ./txt2bin | ./pdm2pcm -f1024000 -d128 | aplay -fS16_LE -c1 -r8000`

Using Cypress FX2LP setup

- Follow instructions provided in the `pdmgrabber` folder for setting up the hardware.
- Grab up to eight microphones in parallel using fx2grabber utility as follows: `fx2grabber -d4 dump.pdm`
- Use `packdata` utility to extract microphone channels and decode them to PCM with:  `packdata.exe 0 < dump.pdm | pdm2pcm -f 2048000 -d128 | aplay -fS16_LE -c1 -r16000` where '0' is the index of the microphone to extract.

WAV → PDM closed loop (`wav2pdm`)

- Input must be mono 16-bit PCM WAV whose sample rate equals `f/d` (e.g. 8000 Hz with `-f1024000 -d128`).
- Encode: `./wav2pdm -f1024000 -d128 < sine8k.wav > sine.pdm`
- Decode raw S16_LE mono: `./pdm2pcm -f1024000 -d128 < sine.pdm > sine_dec.raw`
- Play (after wrapping raw PCM or with ffmpeg):
  `ffmpeg -f s16le -ar 8000 -ac 1 -i sine_dec.raw sine_dec.wav && aplay sine_dec.wav`
  or `aplay -f S16_LE -c 1 -r 8000 sine_dec.raw`
- Automated checks: `python3 test_wav2pdm.py` (build `make` first). Covers rejects, d=128 and d=64 round-trips (correlation, polarity, ~440 Hz).

Stereo (`wav2pdm` + `pdm2pcm -c 2`)

- Input WAV: mono OR stereo 16-bit PCM with `sample_rate == f/d`. Reject ≥3 channels.
- Encode (auto-detects channels from WAV `fmt ` chunk):
  `./wav2pdm -f1024000 -d64 < stereo.wav > stereo.pdm`
- Decode (explicit `-c 2` is required since the PDM stream carries no channel hint):
  `./pdm2pcm -f1024000 -d64 -c2 < stereo.pdm > stereo.raw`
- Wrap raw stereo PCM and play:
  `ffmpeg -f s16le -ar 16000 -ac 2 -i stereo.raw stereo.wav && aplay stereo.wav`
- Stereo test coverage in `python3 test_wav2pdm.py`:
  - `test_pdm2pcm_c_rejects` — bad `-c` values exit 1 with stderr message
  - `test_stereo_silence_bitpattern` — exact 16-byte `95 95 55 55 …` for stereo silence at d=64
  - `test_roundtrip_stereo_{64,128}` — L=440 Hz / R=880 Hz round-trip, per-channel lag-tolerant Pearson + frequency check
