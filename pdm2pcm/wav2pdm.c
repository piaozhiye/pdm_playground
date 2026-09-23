/** \file wav2pdm.c
 * Convert mono/stereo S16 LE WAV to packed PDM for pdm2pcm.
 * Usage: wav2pdm -f <pdm_hz> -d <64|128> [-o 1|2] < in.wav > out.dat
 *        -o 1 : 1st-order Σ-Δ (default)
 *        -o 2 : 2nd-order Σ-Δ (better SNR; modern PDM mics)
 * MSB of each byte is the oldest PDM bit.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void die(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

static uint16_t rd_u16(const uint8_t *p) {
	return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
static uint32_t rd_u32(const uint8_t *p) {
	return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
	       ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
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
			w->bits = rd_u16(fmt + 14);
			extra = sz - 16;
			if (extra && discard_bytes(in, extra) != 0) return -1;
			have_fmt = 1;
		} else if (memcmp(hdr, "data", 4) == 0) {
			if (!have_fmt) return -1;
			w->data_len = sz;
			return 0;
		} else {
			uint32_t skip = sz + (sz & 1u);
			if (discard_bytes(in, skip) != 0) return -1;
		}
	}
}

static void usage(const char *argv0) {
	fprintf(stderr,
		"Usage: %s -f <PDM sampling frequency> -d <64|128> [-o 1|2] < in.wav > out.dat\n"
		"  -o 1 : 1st-order Σ-Δ (default)\n"
		"  -o 2 : 2nd-order Σ-Δ (better SNR; matches modern PDM mics)\n"
		"Example: %s -f1024000 -d128 -o 2 < sine8k.wav > sine.pdm\n",
		argv0, argv0);
}

/* 1st-order Σ-Δ: one integrator. */
static inline int sd1_tick(int32_t *acc, int32_t x) {
	int bit;
	*acc += x;
	if (*acc >= 0) { bit = 1; *acc -= 32768; }
	else           { bit = 0; *acc += 32767; }
	return bit;
}

/* 2nd-order Σ-Δ (CIFB): two integrators in cascade. Both subtract the
 * quantized output each tick. Second-stage state lives in int16_t (NOT
 * int8_t — aliased int32_t would be UB). First-stage stays int32_t because
 * its state range is wider. */
static inline int sd2_tick(int32_t *acc1, int16_t *acc2, int32_t x) {
	int bit;
	*acc1 += x;
	*acc2 += (int16_t)(*acc1 >> 8);  /* scale to prevent overflow in 16-bit */
	if (*acc2 >= 0) {
		bit = 1;
		*acc1 -= 32768;
		*acc2 -= 128;  /* scaled feedback */
	} else {
		bit = 0;
		*acc1 += 32767;
		*acc2 += 127;
	}
	return bit;
}

int main(int argc, char **argv) {
	int opt;
	unsigned int pdm_f = 0, dec = 0, order = 1;
	WavInfo w;
	FILE *in = stdin;
	uint32_t remaining;
	int32_t accL = 0;
	int16_t sd2_acc2_L = 0;
	uint8_t out_byte = 0;
	int bits_in_byte = 0;
	uint8_t sample_le[2];

	while ((opt = getopt(argc, argv, "hf:d:o:")) != -1) {
		switch (opt) {
		case 'h': usage(argv[0]); return 0;
		case 'f': pdm_f = (unsigned int)atoi(optarg); break;
		case 'd':
			dec = (unsigned int)atoi(optarg);
			if (dec != 64 && dec != 128) die("Decimation factor must be 64 or 128");
			break;
		case 'o':
			order = (unsigned int)atoi(optarg);
			if (order != 1 && order != 2) die("Σ-Δ order must be 1 or 2");
			break;
		default: usage(argv[0]); return 1;
		}
	}
	if (pdm_f == 0 || dec == 0) die("Must specify both -f and -d");
	if (pdm_f % dec != 0) die("PDM frequency must be divisible by decimation factor");
	if (parse_wav(in, &w) != 0) die("Invalid or unsupported WAV (need RIFF/WAVE)");
	if (w.format != 1) die("WAV format must be PCM (format tag 1)");
	if (w.channels != 1 && w.channels != 2) {
		fprintf(stderr, "WAV must be 1 or 2 channels (got %u)\n", (unsigned)w.channels);
		exit(1);
	}
	if (w.bits != 16) die("WAV must be 16-bit");
	if (w.sample_rate != pdm_f / dec) {
		fprintf(stderr, "WAV sample rate %u != f/d (%u)\n", w.sample_rate, pdm_f / dec);
		die("Sample rate mismatch");
	}
	if (w.data_len < (uint32_t)(w.channels * 2) ||
	    (w.data_len % (uint32_t)(w.channels * 2)) != 0) {
		fprintf(stderr, "WAV data chunk size %u not aligned to %u-channel S16\n",
			w.data_len, w.channels);
		exit(1);
	}

	remaining = w.data_len;
	if (w.channels == 1) {
		while (remaining >= 2) {
			int16_t x; unsigned int k; int bit;
			if (!read_exact(in, sample_le, 2)) die("Truncated WAV data chunk");
			remaining -= 2;
			x = (int16_t)rd_u16(sample_le);
			for (k = 0; k < dec; k++) {
				if (order == 1) bit = sd1_tick(&accL, x);
				else             bit = sd2_tick(&accL, &sd2_acc2_L, x);
				out_byte = (uint8_t)((out_byte << 1) | (bit & 1));
				bits_in_byte++;
				if (bits_in_byte == 8) {
					if (fwrite(&out_byte, 1, 1, stdout) != 1) die("Write to stdout failed");
					out_byte = 0; bits_in_byte = 0;
				}
			}
		}
	} else {
		int32_t accR = 0;
		int16_t sd2_acc2_R = 0;
		uint8_t sample_le4[4];
		while (remaining >= 4) {
			uint8_t byteL = 0, byteR = 0;
			int bitsL = 0, bitsR = 0;
			int16_t xL, xR; unsigned int k; int bitL, bitR;
			if (!read_exact(in, sample_le4, 4)) die("Truncated WAV data chunk");
			remaining -= 4;
			xL = (int16_t)rd_u16(sample_le4);
			xR = (int16_t)rd_u16(sample_le4 + 2);
			for (k = 0; k < dec; k++) {
				if (order == 1) { bitL = sd1_tick(&accL, xL); bitR = sd1_tick(&accR, xR); }
				else           { bitL = sd2_tick(&accL, &sd2_acc2_L, xL); bitR = sd2_tick(&accR, &sd2_acc2_R, xR); }
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
