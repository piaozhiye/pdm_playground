/**
 * Copyright (c) 2026 piaozhiye <piaozhiye@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 * \file wav2pdm.c
 * Convert mono/stereo S16/S24 LE PCM WAV to packed PDM for pdm2pcm.
 * Usage: wav2pdm -f <pdm_hz> -d <64|128> [-o 1|2] < in.wav > out.dat
 *        -o 1 : 1st-order Σ-Δ (default)
 *        -o 2 : 2nd-order Σ-Δ (experimental bounded-state modulator)
 * MSB of each byte is the oldest PDM bit.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>

static void die(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

static int parse_uint_arg(const char *arg, unsigned int *value) {
	const unsigned char *p = (const unsigned char *)arg;
	char *end = NULL;
	unsigned long parsed;

	if (arg == NULL || *arg == '\0') return -1;
	for (; *p != '\0'; ++p) {
		if (*p < '0' || *p > '9') return -1;
	}
	errno = 0;
	parsed = strtoul(arg, &end, 10);
	if (errno == ERANGE || end == arg || *end != '\0' || parsed > UINT_MAX) {
		return -1;
	}
	*value = (unsigned int)parsed;
	return 0;
}

static uint16_t rd_u16(const uint8_t *p) {
	return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd_u32(const uint8_t *p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static int16_t rd_s16(const uint8_t *p) {
	uint16_t value = rd_u16(p);
	if (value <= INT16_MAX) return (int16_t)value;
	return (int16_t)((int32_t)value - 65536);
}

/* Packed signed 24-bit little-endian sample, sign-extended to int32. */
static int32_t rd_s24(const uint8_t *p) {
	uint32_t value = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	                 ((uint32_t)p[2] << 16);
	if (value & 0x800000u) return (int32_t)value - 0x1000000;
	return (int32_t)value;
}

static int32_t rd_pcm_sample(const uint8_t *p, unsigned int bits) {
	return bits == 16 ? rd_s16(p) : rd_s24(p);
}

static int read_exact(FILE *in, void *buf, size_t n) {
	return fread(buf, 1, n, in) == n;
}
static int discard_bytes(FILE *in, uint32_t n) {
	uint8_t buf[256];
	while (n > 0) {
		size_t chunk = n > sizeof(buf) ? sizeof(buf) : n;
		if (fread(buf, 1, chunk, in) != chunk) return -1;
		n -= (uint32_t)chunk;
	}
	return 0;
}

typedef struct {
	uint16_t format;
	uint16_t channels;
	uint32_t sample_rate;
	uint16_t block_align;
	uint16_t bits;
	uint32_t data_len;
} WavInfo;

static int parse_wav(FILE *in, WavInfo *w) {
	uint8_t riff[12];
	int have_fmt = 0;
	if (!read_exact(in, riff, 12)) return -1;
	if (memcmp(riff, "RIFF", 4) != 0 || memcmp(riff + 8, "WAVE", 4) != 0) return -1;
	for (;;) {
		uint8_t hdr[8];
		uint32_t sz;
		if (!read_exact(in, hdr, 8)) return -1;
		sz = rd_u32(hdr + 4);
		if (memcmp(hdr, "fmt ", 4) == 0) {
			uint8_t fmt[16]; uint32_t extra;
			if (sz < 16) return -1;
			if (!read_exact(in, fmt, 16)) return -1;
			w->format = rd_u16(fmt);
			w->channels = rd_u16(fmt + 2);
			w->sample_rate = rd_u32(fmt + 4);
			w->block_align = rd_u16(fmt + 12);
			w->bits = rd_u16(fmt + 14);
			extra = sz - 16;
			if (extra && discard_bytes(in, extra) != 0) return -1;
			if ((sz & 1u) != 0 && discard_bytes(in, 1) != 0) return -1;
			have_fmt = 1;
		} else if (memcmp(hdr, "data", 4) == 0) {
			if (!have_fmt) return -1;
			w->data_len = sz;
			return 0;
		} else {
			if (sz > UINT32_MAX - (sz & 1u)) return -1;
			uint32_t skip = sz + (sz & 1u);
			if (discard_bytes(in, skip) != 0) return -1;
		}
	}
}

static void usage(const char *argv0) {
	fprintf(stderr,
		"Usage: %s -f <PDM sampling frequency> -d <64|128> [-o 1|2] < in.wav > out.dat\n"
		"  input: mono/stereo S16 or S24 LE PCM WAV (detected from the fmt chunk)\n"
		"  -o 1 : 1st-order Σ-Δ (default)\n"
		"  -o 2 : 2nd-order Σ-Δ (bounded experimental modulator)\n"
		"Example: %s -f1024000 -d128 -o 2 < sine8k.wav > sine.pdm\n",
		argv0, argv0);
}

#define STATE_LIMIT (INT64_C(1) << 30)

static int64_t clamp_state(int64_t value) {
	if (value > STATE_LIMIT) return STATE_LIMIT;
	if (value < -STATE_LIMIT) return -STATE_LIMIT;
	return value;
}

static int64_t arithmetic_shift_right_8(int64_t value) {
	if (value >= 0) return value / 256;
	return -(((-value) + 255) / 256);
}

/* 1st-order Σ-Δ: one integrator. full_scale is the PCM input range, so
 * 16-bit sources use 32768 and 24-bit sources use 8388608. */
static inline int sd1_tick(int64_t *acc, int32_t x, int32_t full_scale) {
	int bit;
	*acc = clamp_state(*acc + x);
	if (*acc >= 0) {
		bit = 1;
		*acc = clamp_state(*acc - full_scale);
	} else {
		bit = 0;
		*acc = clamp_state(*acc + (full_scale - 1));
	}
	return bit;
}

/* 2nd-order Σ-Δ (CIFB): two integrators in cascade. The second stage keeps
 * its 1/256 scale relative to the first, so second_scale = full_scale / 256.
 * Both states are bounded explicitly so malformed or extreme input has
 * defined behavior. */
static inline int sd2_tick(int64_t *acc1, int64_t *acc2, int32_t x,
                           int32_t full_scale, int32_t second_scale) {
	int bit;
	*acc1 = clamp_state(*acc1 + x);
	*acc2 = clamp_state(*acc2 + arithmetic_shift_right_8(*acc1));
	if (*acc2 >= 0) {
		bit = 1;
		*acc1 = clamp_state(*acc1 - full_scale);
		*acc2 = clamp_state(*acc2 - second_scale);
	} else {
		bit = 0;
		*acc1 = clamp_state(*acc1 + (full_scale - 1));
		*acc2 = clamp_state(*acc2 + (second_scale - 1));
	}
	return bit;
}

int main(int argc, char **argv) {
	int opt;
	unsigned int pdm_f = 0, dec = 0, order = 1;
	WavInfo w = {0};
	FILE *in = stdin;
	uint32_t remaining;
	int64_t accL = 0;
	int64_t sd2_acc2_L = 0;
	uint8_t out_byte = 0;
	int bits_in_byte = 0;
	uint8_t sample_le[3];
	uint32_t bytes_per_sample;
	uint32_t frame_bytes;
	int32_t full_scale;
	int32_t second_scale;


	while ((opt = getopt(argc, argv, "hf:d:o:")) != -1) {
		switch (opt) {
		case 'h': usage(argv[0]); return 0;
		case 'f':
			if (parse_uint_arg(optarg, &pdm_f) != 0) die("Invalid PDM frequency");
			break;
		case 'd':
			if (parse_uint_arg(optarg, &dec) != 0 ||
			    (dec != 64 && dec != 128)) {
				die("Decimation factor must be 64 or 128");
			}
			break;
		case 'o':
			if (parse_uint_arg(optarg, &order) != 0 ||
			    (order != 1 && order != 2)) {
				die("Σ-Δ order must be 1 or 2");
			}
			break;
		default: usage(argv[0]); return 1;
		}
	}
	if (optind != argc) die("Unexpected positional argument");
	if (pdm_f == 0 || dec == 0) die("Must specify both -f and -d");
	if (pdm_f % dec != 0) die("PDM frequency must be divisible by decimation factor");
	if (parse_wav(in, &w) != 0) die("Invalid or unsupported WAV (need RIFF/WAVE)");
	if (w.format != 1) die("WAV format must be PCM (format tag 1)");
	if (w.channels != 1 && w.channels != 2) {
		fprintf(stderr, "WAV must be 1 or 2 channels (got %u)\n", (unsigned)w.channels);
		exit(1);
	}
	if (w.bits != 16 && w.bits != 24) die("WAV must be 16-bit or 24-bit");
	if (w.sample_rate != pdm_f / dec) {
		fprintf(stderr, "WAV sample rate %u != f/d (%u)\n", w.sample_rate, pdm_f / dec);
		die("Sample rate mismatch");
	}
	/* The Σ-Δ thresholds follow the source range so a 24-bit WAV is not
	 * driven as if it were 16-bit. */
	bytes_per_sample = w.bits / 8;
	frame_bytes = (uint32_t)w.channels * bytes_per_sample;
	/* A header that disagrees with itself about the frame width would make
	 * us walk the data chunk on the wrong frame boundaries, so reject it
	 * instead of silently resampling from a shifted offset. */
	if (w.block_align != frame_bytes) {
		fprintf(stderr, "WAV block_align %u != channels x bits/8 (%u)\n",
			w.block_align, frame_bytes);
		exit(1);
	}
	full_scale = (int32_t)(1u << (w.bits - 1));
	second_scale = full_scale / 256;
	if (w.data_len < frame_bytes || (w.data_len % frame_bytes) != 0) {
		fprintf(stderr, "WAV data chunk size %u not aligned to %u-channel S%u\n",
			w.data_len, w.channels, w.bits);
		exit(1);
	}

	remaining = w.data_len;
	if (w.channels == 1) {
		while (remaining >= bytes_per_sample) {
			int32_t x; unsigned int k; int bit;
			if (!read_exact(in, sample_le, bytes_per_sample)) die("Truncated WAV data chunk");
			remaining -= bytes_per_sample;
			x = rd_pcm_sample(sample_le, w.bits);
			for (k = 0; k < dec; k++) {
				if (order == 1) bit = sd1_tick(&accL, x, full_scale);
				else             bit = sd2_tick(&accL, &sd2_acc2_L, x, full_scale, second_scale);
				out_byte = (uint8_t)((out_byte << 1) | (bit & 1));
				bits_in_byte++;
				if (bits_in_byte == 8) {
					if (fwrite(&out_byte, 1, 1, stdout) != 1) die("Write to stdout failed");
					out_byte = 0; bits_in_byte = 0;
				}
			}
		}
	} else {
		int64_t accR = 0;
		int64_t sd2_acc2_R = 0;
		uint8_t frame_le[6];
		while (remaining >= frame_bytes) {
			uint8_t byteL = 0, byteR = 0;
			int bitsL = 0, bitsR = 0;
			int32_t xL, xR; unsigned int k; int bitL, bitR;
			if (!read_exact(in, frame_le, frame_bytes)) die("Truncated WAV data chunk");
			remaining -= frame_bytes;
			xL = rd_pcm_sample(frame_le, w.bits);
			xR = rd_pcm_sample(frame_le + bytes_per_sample, w.bits);
			for (k = 0; k < dec; k++) {
				if (order == 1) {
					bitL = sd1_tick(&accL, xL, full_scale);
					bitR = sd1_tick(&accR, xR, full_scale);
				} else {
					bitL = sd2_tick(&accL, &sd2_acc2_L, xL, full_scale, second_scale);
					bitR = sd2_tick(&accR, &sd2_acc2_R, xR, full_scale, second_scale);
				}
				byteL = (uint8_t)((byteL << 1) | (bitL & 1)); bitsL++;
				if (bitsL == 8) {
					if (fwrite(&byteL, 1, 1, stdout) != 1) die("Write to stdout failed");
					byteL = 0; bitsL = 0;
				}
				byteR = (uint8_t)((byteR << 1) | (bitR & 1)); bitsR++;
				if (bitsR == 8) {
					if (fwrite(&byteR, 1, 1, stdout) != 1) die("Write to stdout failed");
					byteR = 0; bitsR = 0;
				}
			}
			if (bitsL > 0) { byteL = (uint8_t)(byteL << (8 - bitsL));
				if (fwrite(&byteL, 1, 1, stdout) != 1) die("Write to stdout failed"); }
			if (bitsR > 0) { byteR = (uint8_t)(byteR << (8 - bitsR));
				if (fwrite(&byteR, 1, 1, stdout) != 1) die("Write to stdout failed"); }
		}
	}
	if (fflush(stdout) != 0) die("Flush stdout failed");
	return 0;
}
