#include "utility.h"

void yellow() { printf("\033[1;33m"); }
void reset() { printf("\033[0m"); }

void evolve( const double* matrix, double* matrix_new, const int* rows, const int rank, const int N) {
  //This will be a row dominant program.
  for (int i = 1 ; i <= rows[rank]; i++)
    for (int j = 1; j <= N; j++)
      matrix_new[ ( i * ( N + 2 ) ) + j ] = ( 0.25 ) *
        ( matrix[ ( ( i - 1 ) * ( N + 2 ) ) + j ] +
          matrix[ ( i * ( N + 2 ) ) + ( j + 1 ) ] +
          matrix[ ( ( i + 1 ) * ( N + 2 ) ) + j ] +
          matrix[ ( i * ( N + 2 ) ) + ( j - 1 ) ] );
}

void save_gnuplot( double* M, const int N, const int rank ) {

  if (rank == 0) {
    const double h = 0.1;
    FILE *file;
    file = fopen( "solution.dat", "w" );

    for (int i = 0; i < N + 2; i++)
      for (int j = 0; j < N + 2; j++)
        fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[ ( i * ( N + 2 ) ) + j ] );

    fclose( file );
  }
}

// A Simple timer for measuring the walltime
double seconds() {
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec) / 1000000.0;
    return sec;
}

void initCounts(const int n_loc, const int wsz, const int rest, int* rows, int* offset) {
  for (int i = 0; i < wsz; i++) rows[i] = (i < rest) ? n_loc + 1 : n_loc; // first processes get the remainder
  offset[0] = 0;
  for (int i = 1; i < wsz; i++) offset[i] = offset[i-1] + rows[i-1];
}

void initMatrix(const int rank, const int wsz, const int N,
                const int* rows, const int* offset, double* matrix) {
  memset( matrix, 0, (rows[rank] + 2) * (N + 2) * sizeof(double) ); // set to zero to check for bugs later
  // fill initial values
  for (int i = 1; i <= rows[rank]; i++) // not the first row
    for (int j = 1; j < N + 1; j++) // not the first and last column either
      matrix[ ( i * ( N + 2 ) ) + j ] = 0.5;

  const double increment = 100.0 / ( N + 1 );  // set up borders

  for (int i = 1; i <= rows[rank]; i++) matrix[i * (N + 2)] = (i + offset[rank]) * increment;

  // The last one fills the lower row
  if (rank == wsz - 1) {
    for (int i = 1; i <= N + 1; ++i) matrix[(rows[rank] + 1) * (N + 2) + (N + 1 - i)] = i * increment;
  }
}

void exchangeRows(double* matrix, const int* rows, const int N, const int rank, const int above, const int below) {
    // Exchange upper row
    MPI_Sendrecv(matrix + N + 2, N + 2, // Send the first (actual) row (the second)
                 MPI_DOUBLE, above, 0, // to the process above
                 matrix + (rows[rank] + 1 ) * (N + 2), N + 2, // Receive it and put it in the lower ghost row (last row)
                 MPI_DOUBLE, below, 0, // from the process below (same tag)
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Exchange lower row
    MPI_Sendrecv(matrix + rows[rank] * (N + 2), N + 2, // Send the last (actual) row (the second to last)
                 MPI_DOUBLE, below, 0, // to the process below
                 matrix, N + 2, // Receive the row and put it in the upper ghost row (the first row)
                 MPI_DOUBLE, above, 0, // from the process above (same tag)
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void printMatrix(const double* M, const int rows, const int cols) {
  for (int i = 0; i < rows; i++ ) {
      for (int j = 0; j < cols; j++ ) {
          fprintf( stdout, "%.3g ", M[i*cols + j] );
      }
      fprintf( stdout, "\n");
  }
}

// without ghost rows
void printCalls(const int wsz, const int rank, const int N, const int* rows, double* matrix) {
  // Print in order
  if (rank == 0) {
      yellow(); printf("Rank %d:\n", rank); reset();
      printMatrix(matrix, rows[rank] + 1, N + 2);
      for (int count = 1; count < wsz; count++) {
          int n_rows = (count == wsz - 1) ? rows[count] + 1 : rows[count];
          MPI_Recv(matrix + N + 2, n_rows * (N + 2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          yellow(); printf("Rank %d:\n", count); reset();
          printMatrix(matrix + N + 2, n_rows, N + 2);
      }
  } else {
        int n_rows = (rank == wsz - 1) ? rows[rank] + 1 : rows[rank];
        MPI_Send(matrix + N + 2, n_rows * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
  }
}

/*
// with ghost rows
void printCalls(const int wsz, const int rank, const int N, const int* rows, double* matrix) {
  // Print in order
  if (rank == 0) {
      yellow(); printf("Rank %d:\n", rank); reset();
      printMatrix(matrix, rows[rank] + 2, N + 2);
      for (int count = 1; count < wsz; count++) {
          MPI_Recv(matrix, (rows[count] + 2) * (N + 2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          yellow(); printf("Rank %d:\n", count); reset();
          printMatrix(matrix, rows[count] + 2, N + 2);
      }
  } else MPI_Send(matrix, (rows[rank] + 2) * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
}
*/
