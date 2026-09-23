all: pdm2pcm txt2bin packdata bin2txt wav2pdm

CFLAGS = -Wall -O3 -I.

pdm2pcm: pdm2pcm.c OpenPDMFilter.c

txt2bin: txt2bin.c

packdata: packdata.c

bin2txt: bin2txt.c

wav2pdm: wav2pdm.c

clean:
	rm -rf pdm2pcm txt2bin packdata bin2txt wav2pdm *~
