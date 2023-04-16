CC=mpicc
NVCC=nvcc

CFLAGS=-Wall -O3

OPENBLAS=${OPENBLAS_HOME}

TARGET=prod.x

SOURCES=MPI_MAT_MUL.c utility.c
OBJECTS=$(SOURCES:.c=.o)

ifeq ($(dgemm),1)
    LDFLAGS=-DUSE_CBLAS -I $(OPENBLAS)/include/ -L $(OPENBLAS)/lib -lopenblas
endif

ifeq ($(gpu),1)
    LDFLAGS+=-DUSE_GPU
    OBJECTS+=gpu.o
    LDLIBS+=-L${CUDA_HOME}/lib64/ -lcudart -lcublas
endif

all: $(TARGET)

%.o: %.c
        $(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

gpu.o: gpu.cu
        $(NVCC) -c $< -o $@

$(TARGET): $(OBJECTS)
        $(CC) $(LDFLAGS) -o $(TARGET) $(OBJECTS) $(LDLIBS)

clean:
        rm -f $(OBJECTS) $(TARGET) gpu.o
