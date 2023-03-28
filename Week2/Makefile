CC = mpicc
CFLAGS = -O3
ifdef USE_BLAS
CFLAGS += -DUSE_BLAS
LIBS = -lblas
endif

MPI_MAT_MUL: MPI_MAT_MUL.o utility.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

MPI_MAT_MUL.o: MPI_MAT_MUL.c
	$(CC) $(CFLAGS) -c $<

utilities.o: utility.c utility.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f MPI_MAT_MUL MPI_MAT_MUL.o utility.o