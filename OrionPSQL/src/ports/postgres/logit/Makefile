PG_INC=$(PGHOME)/include/server/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg
pg: dense clean

dense:
	$(CC) $(CFLAGS) -I$(PG_INC) -c logit.c -o logit.o
	$(CC) $(LDFLAGS) -o logit.so logit.o
	cp logit.so $(PGHOME)/lib/dense-logit-shmem.so
	cp logit.so $(PGHOME)/lib/dense-logit-shmem.so

clean:
	rm *.o *.so
