#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

void save_gnuplot( double *, const int, const int ); // save matrix to file

void evolve( double *, double *, int); // evolve Jacobi

double seconds( void ); // return the elapsed

void initMatrix(const int, const int, const int, const int*, const int*, double*);

void printMatrix(const double*, const int, const int); // print a matrix

void printCalls(const int, const int, const int, const int*, double*);
