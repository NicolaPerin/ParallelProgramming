CC := mpicc
CFLAGS := -Wall -O3

SRC := Obi_wan_Jacobi.c utility.c
OBJ := $(SRC:.c=.o)

# Default target
all: $(OBJ)
	$(CC) $(CFLAGS) -o Obi_wan_Jacobi $(OBJ)

# Target for compiling without OpenACC
$(OBJ): %.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Target for compiling with OpenACC
acc: CFLAGS += -DOPENACC -acc -Minfo=accel -ta=tesla
acc: all

# Clean target
clean:
	rm -f Obi_wan_Jacobi $(OBJ)
