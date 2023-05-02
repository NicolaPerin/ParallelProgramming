#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#ifdef ACC
#include <openacc.h>
#endif

void reset( void ); void red( void ); void green( void ); void yellow( void );

void save_gnuplot(double *, const int, const int, const int, const int*, const int*); // save matrix to file

void exchangeRows(double*, const int*, const int, const int, const int, const int); // MPI_Sendrecv()

void evolve(const double*, double*, const int*, const int, const int); // evolve Jacobi

void Jacobi(double*, double*, const int*, const int, const int, const int, const int, double);

void initCounts(const int, const int, const int, int*, int*); // determine nr of rows and offset

void initMatrix(const int, const int, const int, const int*, const int*, double*); // initial conditions

void printMatrix(const double*, const int, const int); // print a matrix

void printCalls(const int, const int, const int, const int*, double*); // send to rank 0 in order and write to terminal
