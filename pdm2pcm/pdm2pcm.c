/** \file pdm2pcm.c
 * \author david.siorpaes@gmail.com
 *
 * Uses OpenPDM library to decode pdm data coming from standard input
 * and sends PCM data to standard output
 * Example usage:
 *
 * bzcat bellazio.txt.bz2 | ./txt2bin | ./pdm2pcm -f 1024000 -d128 | aplay -fS16_LE -c1 -r8000
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include "OpenPDMFilter.h"

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

int main(int argc, char** argv)
{
	int opt;
	ssize_t ret;
	size_t dataCount;
	int finished = 0;
	unsigned int pdmSamplingF, decimationF, pcmSamplingF, pdmBufLen, pcmBufLen;
	unsigned int channels = 1;
	uint8_t* pdmBuf;
	int16_t* pcmBuf;
	TPDMFilter_InitStruct filter;

	/* Get user options */
	pdmSamplingF = decimationF = 0;
	while((opt = getopt (argc, argv, "hf:d:c:")) != -1){
		switch(opt){
			case 'h':
				printf("%s -h(elp) -f <PDM sampling frequency> -d <64|128> [-c <1|2>]\n", argv[0]);
				printf("  -c 1: mono (default)\n");
				printf("  -c 2: stereo\n");
				printf("Example usage: bzcat bellazio.txt.bz2 | ./txt2bin | "
				       "./pdm2pcm -f 1024000 -d128 | aplay -fS16_LE -c1 -r8000\n");
				exit(0);
				break;
				
			case 'f':
				if (parse_uint_arg(optarg, &pdmSamplingF) != 0) {
					fprintf(stderr, "Invalid PDM sampling frequency\n");
					exit(1);
				}
				break;

			case 'd':
				if (parse_uint_arg(optarg, &decimationF) != 0 ||
				   (decimationF != 64 && decimationF != 128)) {
					fprintf(stderr, "Decimation factor must be 64 or 128\n");
					exit(1);
				}
				break;

			case 'c':
				if (parse_uint_arg(optarg, &channels) != 0 ||
				   (channels != 1 && channels != 2)) {
					fprintf(stderr, "Channel count must be 1 or 2 (got %u)\n", channels);
					exit(1);
				}
				break;

			case '?':
				if(optopt == 'f' || optopt == 'd' || optopt == 'c'){
					fprintf(stderr, "Option -%c requires argument\n", optopt);
					exit(1);
				}
				exit(1);
				break;
				
			default:
				break;
		}
	}

	if (optind != argc) {
		fprintf(stderr, "Unexpected positional argument\n");
		exit(1);
	}
	if(decimationF == 0 || pdmSamplingF == 0){
		fprintf(stderr, "Must specify both PDM sampling frequency and decimation factor\n");
		exit(1);
	}
	if (pdmSamplingF % (1000u * decimationF) != 0) {
		fprintf(stderr, "PDM frequency must be divisible by 1000 * decimation factor\n");
		exit(1);
	}

	pcmSamplingF = pdmSamplingF/decimationF;
	if (pcmSamplingF > UINT16_MAX) {
		fprintf(stderr, "PCM sampling rate is not supported by the filter API\n");
		exit(1);
	}

	/* Allocate buffers to contain 1ms worth of data (per channel for PDM, total for PCM). */
	pdmBufLen = pdmSamplingF/1000;
	pcmBufLen = pdmBufLen/decimationF;
	if (pcmBufLen == 0 || (pdmBufLen % 8) != 0) {
		fprintf(stderr, "PDM frequency does not produce complete 1ms decimation blocks\n");
		exit(1);
	}

	const size_t pdmBlockBytes = (size_t)(pdmBufLen / 8) * channels;
	const size_t pcmBlockBytes = sizeof(int16_t) * (size_t)pcmBufLen * channels;

	pdmBuf = malloc(pdmBlockBytes);
	if(pdmBuf == NULL){
		fprintf(stderr, "Cannot allocate memory\n");
		exit(EXIT_FAILURE);
	}

	pcmBuf = malloc(pcmBlockBytes);
	if(pcmBuf == NULL){
		free(pdmBuf);
		fprintf(stderr, "Cannot allocate memory\n");
		exit(EXIT_FAILURE);
	}
	
	/* Initialize Open PDM library */
	filter.Fs = pcmSamplingF;
	filter.nSamples = pcmBufLen;
	filter.LP_HZ = pcmSamplingF/2;
	filter.HP_HZ = 10;
	filter.In_MicChannels = channels;
	filter.Out_MicChannels = channels;
	filter.Decimation = decimationF;
	filter.MaxVolume = 16;
	Open_PDM_Filter_Init(&filter);

	while(finished == 0){
		/* Grab one complete 1ms PDM block from stdin. */
		dataCount = 0;
		while(dataCount < pdmBlockBytes){
			ret = read(STDIN_FILENO, pdmBuf + dataCount, pdmBlockBytes - dataCount);
			if(ret < 0){
				if(errno == EINTR) continue;
				fprintf(stderr, "Error reading from STDIN: %s\n", strerror(errno));
				exit(EXIT_FAILURE);
			}
			if(ret == 0){
				finished = 1;
				break;
			}
			dataCount += (size_t)ret;
		}

		if(dataCount == 0) break;
		if(dataCount != pdmBlockBytes){
			fprintf(stderr, "Incomplete PDM block: got %zu of %zu bytes\n",
				dataCount, pdmBlockBytes);
			exit(EXIT_FAILURE);
		}

		/* Decode PDM. Oldest PDM bit is MSB. */
		if(decimationF == 64)
			Open_PDM_Filter_64(pdmBuf, pcmBuf, 1, &filter);
		else
			Open_PDM_Filter_128(pdmBuf, pcmBuf, 1, &filter);

		/* Emit PCM decoded data to stdout. */
		dataCount = 0;
		while(dataCount < pcmBlockBytes){
			ret = write(STDOUT_FILENO, pcmBuf + dataCount, pcmBlockBytes - dataCount);
			if(ret < 0){
				if(errno == EINTR) continue;
				fprintf(stderr, "Error writing to STDOUT: %s\n", strerror(errno));
				exit(EXIT_FAILURE);
			}
			if(ret == 0){
				fprintf(stderr, "Error writing to STDOUT: zero-byte write\n");
				exit(EXIT_FAILURE);
			}
			dataCount += (size_t)ret;
		}
	}

	return 0;
}
