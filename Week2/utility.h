#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

void printMatrix(const double*, const int, const int);

void init_row_counts(int *, const int, const int, const int, const int);

void init_displ_B(int *, const int *, const int , const int );

void init_elem_counts(int *, const int *, const int, const int);

void init_displ_B_col(int *, const int *, const int);

void init_A(double *, const int, const int, const int, const int);

void init_B(double *, const int, const int, const int, const int);

void matrixMultiply(double *, const double *, const double *, const int *, 
                    const int *, const int, const int, const int);