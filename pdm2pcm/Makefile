all: pdm2pcm txt2bin packdata bin2txt wav2pdm

CFLAGS = -Wall -O3 -I.

pdm2pcm: pdm2pcm.c OpenPDMFilter.c OpenPDMFilter.h

txt2bin: txt2bin.c

packdata: packdata.c

bin2txt: bin2txt.c

wav2pdm: wav2pdm.c

test: all
	python3 test_wav2pdm.py

clean:
	rm -rf pdm2pcm txt2bin packdata bin2txt wav2pdm *~
