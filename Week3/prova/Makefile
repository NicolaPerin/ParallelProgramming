EXE = prod

SRC = p.cu

OBJ = $(SRC:.cu=.o)

all: $(EXE)

%.o: %.cu
        nvcc -I/cineca/prod/opt/compilers/cuda/11.0/none/include -c $< -o $@

$(EXE): $(OBJ)
        mpicxx -L/cineca/prod/opt/compilers/cuda/11.0/none/lib64 -lcudart -lcublas -o $@ $^

clean:
        rm -f $(EXE) $(OBJ)
