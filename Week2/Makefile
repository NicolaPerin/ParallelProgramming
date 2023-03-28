CC=mpicc

CFLAGS=-Wall -O3

OPENBLAS=${OPENBLAS_HOME}

TARGET=prod.x

SOURCES=MPI_MAT_MUL.c utility.c

OBJECTS=$(SOURCES:.c=.o)

ifeq ($(USE_BLAS),1)
    LDFLAGS=-DUSE_BLAS -I $(OPENBLAS)/include/ -L $(OPENBLAS)/lib -lopenblas
else
    LDFLAGS=-I $(OPENBLAS)/include/ -L $(OPENBLAS)/lib -lopenblas
endif

all: $(TARGET)

%.o: %.c
        $(CC) $(CFLAGS) $(LDFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
        $(CC) $(LDFLAGS) -o $(TARGET) $(OBJECTS)

clean:
        rm -f $(OBJECTS) $(TARGET)
