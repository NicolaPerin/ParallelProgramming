# inputs for the executables

dim = 10
it  = 100
r = 3
c = 3

CC=mpicc
CFLAGS=-O3 -Wall

TARGET=jacobi.x

SOURCES=Obi_Wan_Jacobi.c utility.c
OBJECTS=$(SOURCES:.c=.o)

all: $(TARGET)

run: clean all
	./$(EXE) $(dim) $(it) $(r) $(c)

%.o: %.c
		$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJECTS)
		$(CC) -o $(TARGET) $(OBJECTS)

clean:
	rm -f $(OBJECTS) $(TARGET) solution.dat 

plot:
	@gnuplot -p plot.plt

.PHONY: clean plot all
