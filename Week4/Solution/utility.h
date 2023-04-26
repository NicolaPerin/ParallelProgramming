#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

void yellow( void );
void reset( void );

void save_gnuplot( double *, const int, const int ); // save matrix to file

void exchangeRows(double*, const int*, const int, const int, const int, const int);

void evolve( const double*, double*, const int*, const int, const int); // evolve Jacobi

double seconds( void ); // return the elapsed time

void initCounts(const int, const int, const int, int*, int*);

void initMatrix(const int, const int, const int, const int*, const int*, double*);

void printMatrix(const double*, const int, const int); // print a matrix

void printCalls(const int, const int, const int, const int*, double*);
