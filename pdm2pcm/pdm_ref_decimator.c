/**
 * SPDX-License-Identifier: Apache-2.0
 *
 * pdm_ref_decimator.c - 参考 PDM 抽取器，用于量化对比 OpenPDMFilter。
 *
 * 为什么需要它：OpenPDMFilter 用固定 3 相 LUT（SINCN=3），核长上限
 * 3 * decimation，带外抑制很弱。要判断 24-bit 输出到底能不能兑现收益，
 * 必须先知道"链路误差地板"有多低，而这个地板主要由抽取核决定。
 *
 * 本工具是测量仪器，不是 OpenPDMFilter 的替代品。它实现可配置的
 * Kaiser 窗 sinc 低通，核长和 beta 都能调，用来分离三个变量：
 *   抽取核质量 / 抽取率 d / 输出位深
 *
 * 输入布局与 pdm2pcm 完全一致（已用非对称双声道源验证）：
 *   每个输出样本占 (d/8) * channels 字节，声道在字节级交错，
 *   即 L0 R0 L1 R1 ...（d=64 时每声道 8 字节）
 *
 * 输出：1 ms block 为单位，16-bit 为 S16_LE，24-bit 为 packed S24_LE。
 */
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_TAPS_PER_PHASE 512
#define MAX_LEN (128 * MAX_TAPS_PER_PHASE)

static void fail(const char *msg) {
	fprintf(stderr, "pdm_ref_decimator: %s\n", msg);
	exit(1);
}

static int parse_long(const char *arg, long *out) {
	char *end = NULL;
	long value;
	if (arg == NULL || *arg == '\0') return -1;
	value = strtol(arg, &end, 10);
	if (end == arg || *end != '\0') return -1;
	*out = value;
	return 0;
}

static int parse_double(const char *arg, double *out) {
	char *end = NULL;
	double value;
	if (arg == NULL || *arg == '\0') return -1;
	value = strtod(arg, &end);
	if (end == arg || *end != '\0') return -1;
	*out = value;
	return 0;
}

static double bessel_i0(double x) {
	double sum = 1.0, term = 1.0;
	int k;
	for (k = 1; k < 80; k++) {
		term *= (x / 2.0) * (x / 2.0) / ((double)k * (double)k);
		sum += term;
		if (term < sum * 1e-17) break;
	}
	return sum;
}

/*
 * Kaiser 窗 sinc 低通。
 *
 * 截止频率 = 0.5 / decimation 个输入采样周期。
 * decimate 倍之后输出采样率是输入的 1/decimation，所以输入奈奎斯特
 * 对应 0.5/decimation，取它正好让输出奈奎斯特落在通带边缘。
 *
 * 推导：理想 sinc 的冲激响应是 sin(2*pi*fc*n)/(2*pi*fc*n)，fc 单位是
 * cycles/sample。写成 np.sinc 形式即 sinc(2*fc*n)，因此这里用
 * sin(pi*fc*2*x) 形式，fc = 0.5/decimation。
 *
 * 注意：这里曾经写成 0.25/decimation，等效截止只有 12 kHz，会把 3 kHz
 * 以上的信号整段削掉，测出的误差毫无意义。修正后 d=64 和 d=128 的截止
 * 都是 24 kHz。
 */
static void design_kernel(long decimation, long taps_per_phase, double beta,
                          double *h, long *len_out) {
	long len = decimation * taps_per_phase;
	long k;
	double center = (double)(len - 1) / 2.0;
	double i0beta = bessel_i0(beta);
	double sum = 0.0;
	/*
	 * 截止频率 = 0.5 / decimation 个输入采样周期。
	 * decimate 倍之后输出采样率是输入的 1/decimation，输出奈奎斯特
	 * 正好对应输入的 0.5/decimation。
	 */
	const double fc = 0.5 / (double)decimation;

	for (k = 0; k < len; k++) {
		double x = (double)k - center;
		double arg = 2.0 * M_PI * fc * x;
		double sinc, ratio;
		if (fabs(x) < 1e-12) {
			sinc = 1.0;
		} else {
			sinc = sin(arg) / arg;
		}
		ratio = 2.0 * (double)k / (double)(len - 1) - 1.0;
		h[k] = sinc * (bessel_i0(beta * sqrt(1.0 - ratio * ratio)) / i0beta);
		sum += h[k];
	}
	/* 归一化到单位直流增益，这样下游可以直接做最小二乘拟合 */
	for (k = 0; k < len; k++) h[k] /= sum;
	*len_out = len;
}

int main(int argc, char **argv) {
	long pdm_rate = 0, decimation = 0, channels = 1, bits = 16;
	long taps_per_phase = 16;
	double beta = 12.0, volume = 1.0, hp_hz = 0.0;
	int opt;

	long pcm_rate, block_bits, block_bytes, block_frames, len, ring;
	long i, c, n;
	double *kernel = NULL;
	int32_t *kernel_q = NULL;
	double kernel_scale = 1.0;
	uint8_t *block = NULL;
	int8_t *bits_buf = NULL;
	int16_t *out16 = NULL;
	int32_t *out24 = NULL;
	uint8_t *packed = NULL;
	long consumed = 0, produced = 0;

	while ((opt = getopt(argc, argv, "f:d:c:b:t:z:v:h:")) != -1) {
		switch (opt) {
		case 'f': if (parse_long(optarg, &pdm_rate)) fail("bad -f"); break;
		case 'd': if (parse_long(optarg, &decimation)) fail("bad -d"); break;
		case 'c': if (parse_long(optarg, &channels)) fail("bad -c"); break;
		case 'b': if (parse_long(optarg, &bits)) fail("bad -b"); break;
		case 't': if (parse_long(optarg, &taps_per_phase)) fail("bad -t"); break;
		case 'z': if (parse_double(optarg, &beta)) fail("bad -z"); break;
		case 'v': if (parse_double(optarg, &volume)) fail("bad -v"); break;
		case 'h': if (parse_double(optarg, &hp_hz)) fail("bad -h"); break;
		default:
			fprintf(stderr,
				"usage: pdm_ref_decimator -f <pdm_hz> -d <64|128> [-c 1|2]\n"
				"                          [-b 16|24] [-t taps_per_phase]\n"
				"                          [-z kaiser_beta] [-v volume] [-h hp_hz]\n");
			return 1;
		}
	}

	if (pdm_rate <= 0 || decimation <= 0) fail("need -f and -d");
	if (decimation != 64 && decimation != 128) fail("-d must be 64 or 128");
	if (channels != 1 && channels != 2) fail("-c must be 1 or 2");
	if (bits != 16 && bits != 24) fail("-b must be 16 or 24");
	if (taps_per_phase < 2 || taps_per_phase > MAX_TAPS_PER_PHASE)
		fail("-t out of range");
	if (beta < 0.0 || beta > 20.0) fail("-z must be 0..20");

	pcm_rate = pdm_rate / decimation;
	block_bits = pdm_rate / 1000;
	block_frames = pcm_rate / 1000;
	if (pdm_rate % 1000 != 0 || block_bits % 8 != 0 || block_frames <= 0)
		fail("rate not aligned to whole 1 ms blocks");
	block_bytes = (block_bits / 8) * channels;

	kernel = malloc(sizeof(double) * (size_t)(decimation * taps_per_phase));
	if (!kernel) fail("out of memory");
	design_kernel(decimation, taps_per_phase, beta, kernel, &len);

	/* 量化到 int32：最大抽头用满大部分量程，再记住换算因子。 */
	{
		double peak = 0.0;
		for (i = 0; i < len; i++) {
			if (fabs(kernel[i]) > peak) peak = fabs(kernel[i]);
		}
		if (peak <= 0.0) fail("degenerate kernel");
		kernel_scale = 2147483000.0 / peak;
		kernel_q = malloc(sizeof(int32_t) * (size_t)len);
		if (!kernel_q) fail("out of memory");
		for (i = 0; i < len; i++)
			kernel_q[i] = (int32_t)llround(kernel[i] * kernel_scale);
	}

	/*
	 * 环形缓冲不变量：产出第 m 个输出样本时，需要输入区间
	 * [m*d - len/2, m*d + len/2)。已经写入的输入到 consumed 为止。
	 * 同一个 block 内先写满整块再产出，所以最老需要的样本距
	 * consumed 最多 (block_bits + len/2)。ring 取 block_bits + len。
	 */
	ring = block_bits + len;
	bits_buf = calloc((size_t)ring * channels, sizeof(int8_t));
	block = malloc((size_t)block_bytes);
	out16 = malloc(sizeof(int16_t) * (size_t)(block_frames * channels));
	out24 = malloc(sizeof(int32_t) * (size_t)(block_frames * channels));
	packed = malloc((size_t)(block_frames * channels * 3));
	if (!bits_buf || !block || !out16 || !out24 || !packed)
		fail("out of memory");

	for (;;) {
		ssize_t got = read(STDIN_FILENO, block, (size_t)block_bytes);
		if (got == 0) break;
		if (got < 0) {
			if (errno == EINTR) continue;
			fail("read failed");
		}
		if (got != (ssize_t)block_bytes) fail("incomplete 1 ms PDM block");

		/*
		 * 解包：声道在字节级交错（pdm2pcm 的 filter_table 从 data[0],data[2]...
		 * 取 L，从 data[1],data[3]... 取 R），MSB first。
		 *
		 * consumed 是"每声道"的 bit 计数，所以字节 i 对应每声道位置
		 * i / channels，不能直接用 i。
		 */
		for (i = 0; i < block_bytes; i++) {
			int byte;
			long ch_byte = i / channels;
			int ch = (int)(i % channels);
			for (byte = 0; byte < 8; byte++) {
				long pos = consumed + ch_byte * 8 + byte;
				int value = (block[i] >> (7 - byte)) & 1;
				bits_buf[(pos % ring) * channels + ch] = (int8_t)value;
			}
		}
		consumed += (long)block_bytes / channels * 8;

		for (n = 0; n < block_frames; n++) {
			long out_index = n * channels;
			for (c = 0; c < channels; c++) {
				double acc = 0.0;
				/*
				 * 核必须拖在采样时刻之后（因果），不能跨到它前面。
				 *
				 * 若把核心放在采样瞬间，某个 1 ms block 的最后几个输出
				 * 需要读到 consumed 之后的输入；本工具按 block 流式处理，
				 * 那些抽头只能当 0 密度，于是每个 block 末尾都引入一次
				 * 瞬态，实测把残差从 0.6 LSB16 抬到 303 LSB16。
				 *
				 * 取 start = m*d - (len - decimation/2)，则最老读到的
				 * 是 start、最晚读到 m*d + decimation/2 - 1，都在已写入
				 * 的范围内。代价是多了 len - decimation/2 的群延迟，
				 * 下游做延迟对齐即可。
				 */
				long start = (produced + n) * decimation - (len - decimation / 2);
				long k;
				for (k = 0; k < len; k++) {
					long pos = start + k;
					if (pos < 0 || pos >= consumed) continue;  /* 视为 0 密度 */
					if (bits_buf[(pos % ring) * channels + c]) acc += (double)kernel_q[k];
				}
				acc /= kernel_scale;   /* 回到单位直流增益 */
				acc -= 0.5;           /* 密度 0.5 是零电平 */
				acc *= 2.0;           /* 满幅密度摆动 -> ±1 */
				acc *= volume;
				if (bits == 24) {
					long v = lrint(acc * 8388607.0);
					if (v > 8388607) v = 8388607;
					if (v < -8388608) v = -8388608;
					out24[out_index + c] = (int32_t)v;
				} else {
					long v = lrint(acc * 32767.0);
					if (v > 32700) v = 32700;
					if (v < -32700) v = -32700;
					out16[out_index + c] = (int16_t)v;
				}
			}
		}
		produced += block_frames;

		if (bits == 24) {
			for (i = 0; i < block_frames * channels; i++) {
				unsigned int u = (unsigned int)out24[i] & 0xFFFFFFu;
				packed[i * 3 + 0] = (uint8_t)(u & 0xFF);
				packed[i * 3 + 1] = (uint8_t)((u >> 8) & 0xFF);
				packed[i * 3 + 2] = (uint8_t)((u >> 16) & 0xFF);
			}
			if (fwrite(packed, 1, (size_t)(block_frames * channels * 3), stdout)
			    != (size_t)(block_frames * channels * 3)) fail("write failed");
		} else {
			if (fwrite(out16, 1, (size_t)(block_frames * channels * 2), stdout)
			    != (size_t)(block_frames * channels * 2)) fail("write failed");
		}
	}

	free(kernel); free(kernel_q); free(bits_buf);
	free(block); free(out16); free(out24); free(packed);
	(void)hp_hz;  /* 本工具不做高通，测试用 DC 由评估脚本去除 */
	return 0;
}
