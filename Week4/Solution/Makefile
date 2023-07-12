CC = mpicc

CFLAGS = -Wall -Wextra -O3

SOURCES = Obi_Wan_Jacobi.c utility.c

OBJECTS = $(SOURCES:.c=.o)

HEADERS = utility.h

all: $(OBJECTS)
        $(CC) $(CFLAGS) -o Jacobi.x $(OBJECTS)

# OpenACC target
acc: CFLAGS += -DACC -acc -Minfo=accel -fast -ta=tesla
acc: all

plot:
        @gnuplot -p plot.plt

clean:
        rm -f $(OBJECTS) Jacobi.x
