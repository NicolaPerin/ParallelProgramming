FFTW_INC = /m100_work/PROJECTS/spack/spack-0.14/install/linux-rhel7-power9le/gcc-8.4.0/fftw-3.3.8-hwlrarpm6cvjlukhfdowwveb7g7oqwgc/include
FFTW_LIB = /m100_work/PROJECTS/spack/spack-0.14/install/linux-rhel7-power9le/gcc-8.4.0/fftw-3.3.8-hwlrarpm6cvjlukhfdowwveb7g7oqwgc/lib

LIBS = -L$(FFTW_LIB) -lfftw3_mpi -lfftw3 -lm
EXE = fftw_mpi.x

SRCS = $(wildcard *.c)
OBJS = $(SRCS:.c=.o)

CC = mpicc
CFLAGS = -O3 -Wall -Wextra

CORES = 4

all: $(EXE)

$(EXE): $(OBJS)
        $(CC) $^ $(LIBS) -o $@

%.o: %.c
        $(CC) $(CFLAGS) -c $< -o $@

run: flush clean all
        mpirun -np $(CORES) $(EXE)

plot:
        gnuplot animate.plt

flush:
        rm -f *.dat
        rm -f *.gif
        rm -f *.png

clean:
        rm -f $(OBJS) $(EXE)
