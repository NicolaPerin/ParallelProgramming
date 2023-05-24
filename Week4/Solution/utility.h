#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#ifdef ACC
#include <openacc.h>
#endif

void reset( void ); void red( void ); void green( void ); void yellow( void ); void blue( void ); // Colors

//void save_gnuplot1(const double*, const size_t, int*, const int, const int); // Parallel I/O attempt (not working)

void save_gnuplot(const double*, const int, const int, const int, FILE*); // Write to file

void exchangeRows(double*, const int*, const size_t, const int, const int, const int); // MPI_Sendrecv()

void evolve(double*, double*, const int*, const int, const size_t, const size_t); // Grid update

void Jacobi(double*, double*, const int*, const int, const int, const size_t, const int, double*); // Simulation

void initCounts(const size_t, const int, const size_t, int*, int*); // Determine nr of rows and offset

void initMatrix(const int, const int, const int, const int*, const int*, double*); // Initial conditions

void hsvToRgb(float, float, float, int*, int*, int*); // Hue Saturation Value color scale

void getColor(int, int, int, int*, int*, int*); // Determine the rgb values for a given temperature

void printMatrix(const double*, const int, const int, const int, const int); // Print a matrix

void printCalls(const int, const int, const size_t, const int*, double*); // Send to rank 0 in order and print to terminal

void writeCalls(const int, const int, const size_t, const int*, double*); // Send to rank 0 in orfer and write to file
