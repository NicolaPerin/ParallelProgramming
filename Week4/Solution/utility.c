#include "utility.h"

void evolve( double * matrix, double *matrix_new, int N ) {
  //This will be a row dominant program.
  for (int i = 1 ; i <= N; i++)
    for (int j = 1; j <= N; j++)
      matrix_new[ ( i * ( N + 2 ) ) + j ] = ( 0.25 ) * 
	( matrix[ ( ( i - 1 ) * ( N + 2 ) ) + j ] + 
	  matrix[ ( i * ( N + 2 ) ) + ( j + 1 ) ] + 	  
	  matrix[ ( ( i + 1 ) * ( N + 2 ) ) + j ] + 
	  matrix[ ( i * ( N + 2 ) ) + ( j - 1 ) ] ); 
}

void save_gnuplot( double *M, const int N, const int rank ) {

  const double h = 0.1;
  FILE *file;

  file = fopen( "solution.dat", "w" );

  for (int i = 0; i < N + 2; i++)
    for (int j = 0; j < N + 2; j++)
      fprintf(file, "%f\t%f\t%f\n", h * j, -h * i, M[ ( i * ( N + 2 ) ) + j ] );

  fclose( file );
}

// A Simple timer for measuring the walltime
double seconds() {
    struct timeval tmp;
    double sec;
    gettimeofday( &tmp, (struct timezone *)0 );
    sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
    return sec;
}

void initMatrix(const int rank, const int wsz, const int N, 
                const int* ghost_counts, const int* displ, double* matrix) {
  // fill initial values  
  for (int i = 1; i < ghost_counts[rank]; i++) // not the first row
    for (int j = 1; j < N + 1; j++) // not the first and last column either
      matrix[ ( i * ( N + 2 ) ) + j ] = 0.5;
  
  const double increment = 100.0 / ( N + 1 );  // set up borders

  // the block goes from 0 to ghost_counts to accomodate one ghost row above and below
  int offset = (rank > 0) ? displ[rank] - 1 : displ[rank];
  for (int i = 1; i < ghost_counts[rank]; i++)
    matrix[i * (N + 2)] = (i + offset) * increment;

  // The last one fills the lower row
  if (rank == wsz - 1) {
    for (int i = 1; i < N + 1; ++i)
      matrix[((ghost_counts[rank] - 1) * (N + 2)) + (N + 1 - i)] = i * increment;
  }
}

void printMatrix(const double* M, const int rows, const int cols) {
  for (int i = 0; i < rows; i++ ) {
      for (int j = 0; j < cols; j++ ) {
          fprintf( stdout, "%.3g ", M[i*cols + j] );
      }
      fprintf( stdout, "\n");
  }
}

void printCalls(const int wsz, const int rank, const int N, const int* recv_counts, double* matrix) {
  if (wsz > 1) {
    if (rank == 0) {
      printf("RANK 0\n");
      printMatrix(matrix, recv_counts[rank], N + 2); // print from the beginning
      for (int count = 1; count < wsz - 1; count++) {
          MPI_Recv(matrix + N + 2, recv_counts[count] * (N + 2), MPI_DOUBLE, count, count, 
          MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("RANK %d\n", count);
          printMatrix(matrix + N + 2, recv_counts[count], N + 2); // print from the 2nd row
      } MPI_Recv(matrix + N + 2, recv_counts[wsz-1] * (N + 2), MPI_DOUBLE, wsz - 1, wsz - 1, 
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("RANK %d\n", wsz - 1);
        printMatrix(matrix + N + 2, recv_counts[wsz-1], N + 2); // print from the 2nd row   
    } else if (rank == wsz - 1) {
      MPI_Send(matrix + N + 2, recv_counts[rank] * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    } else MPI_Send(matrix + N + 2, recv_counts[rank] * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
  } else printMatrix(matrix, N + 2, N + 2);
}
