PG_INC=$(PGHOME)/include/server/
CFLAGS=-O3 -I../../.. -fpic 
LDFLAGS=-shared
CC=gcc

all: pg
pg: array clean

array:
	$(CC) $(CFLAGS) -I$(PG_INC) -c array.c -o array.o
	$(CC) $(LDFLAGS) -o array.so array.o
	cp array.so $(PGHOME)/lib/bismarck-array.so
	cp array.so $(PGHOME)/lib/bismarck-array.so

clean:
	rm *.o *.so

