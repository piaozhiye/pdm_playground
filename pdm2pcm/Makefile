all: pdm2pcm txt2bin packdata bin2txt wav2pdm pdm_ref_decimator

CFLAGS = -Wall -O3 -I.
LDLIBS = -lm

pdm2pcm: pdm2pcm.c OpenPDMFilter.c OpenPDMFilter.h

txt2bin: txt2bin.c

packdata: packdata.c

bin2txt: bin2txt.c

wav2pdm: wav2pdm.c

# Reference decimator used to measure where the decoder error floor comes
# from. Not part of the shipping path; the tests skip it if it is absent.
pdm_ref_decimator: pdm_ref_decimator.c
	$(CC) $(CFLAGS) $< -o $@ $(LDLIBS)

test: all
	python3 test_wav2pdm.py

clean:
	rm -rf pdm2pcm txt2bin packdata bin2txt wav2pdm pdm_ref_decimator *~
