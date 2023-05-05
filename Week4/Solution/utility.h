#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#ifdef ACC
#include <openacc.h>
#endif

void reset( void ); void red( void ); void green( void ); void yellow( void ); // Colors

void save_gnuplot(double *, const size_t, const size_t, const size_t, const size_t*, const size_t*); // Save matrix to file

void exchangeRows(double*, const size_t*, const size_t, const size_t, const size_t, const size_t); // MPI_Sendrecv()

void evolve(double*, double*, const size_t*, const size_t, const size_t, const size_t); // Grid update

void Jacobi(double*, double*, const size_t*, const size_t, const size_t, const size_t, const size_t, double*); // Simulation

void initCounts(const size_t, const size_t, const size_t, size_t*, size_t*); // Determine nr of rows and offset

void initMatrix(const size_t, const size_t, const size_t, const size_t*, const size_t*, double*); // Initial conditions

void printMatrix(const double*, const size_t, const size_t); // Print a matrix

void printCalls(const size_t, const size_t, const size_t, const size_t*, double*); // Send to rank 0 in order and write to terminal
