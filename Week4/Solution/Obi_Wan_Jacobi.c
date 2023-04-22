#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

void save_gnuplot( double *M, int dim ); // save matrix to file
void evolve( double * matrix, double *matrix_new, int N ); // evolve Jacobi
double seconds( void ); // return the elapsed
void printMatrix(const double* M, const int rows, const int cols); // print a matrix

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int rank, wsz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsz);

  double t_start, t_end, increment; // timing variables
  double *matrix, *matrix_new, *tmp_matrix; // initialize matrix + swap pointer
  int N = 0, iterations = 0, row_peek = 0, col_peek = 0;

  // check on input parameters
  if(argc != 5) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
    return 1;
  }

  N = atoi(argv[1]); // 9
  iterations = atoi(argv[2]);
  row_peek = atoi(argv[3]);
  col_peek = atoi(argv[4]);

  int n_loc = (N + 2) / wsz; // 11 / 3 = 3
  int rest = (N + 2) % wsz; // 11 % 3 = 2
  if (rank == 0) printf("n_loc: %d rest: %d\n", n_loc, rest);
  int* row_counts = (int *) malloc( wsz * sizeof(int) );
  int* recv_counts = (int *) malloc( wsz * sizeof(int) );
  int* displ = (int *) malloc( wsz * sizeof(int) );

  // If rank < rest, the process gets (N + 2) / wsz + 3 rows, otherwise, + 2
  for (int i = 0; i < wsz; i++) { row_counts[i] = (i < rest) ? n_loc + 3 : n_loc + 2; }
  for (int i = 0; i < wsz; i++) recv_counts[i] = row_counts[i] - 2;
  row_counts[0] -= 1; row_counts[wsz - 1] -= 1; // first and last processes only have 1 ghost rows
  displ[0] = 0;
  for (int i = 1; i <= wsz; i++) displ[i] = displ[i-1] + recv_counts[i-1];

  if (rank == 0) {
    printf("matrix size: (int) %d (ext) %d\n", N, N + 2);
    printf("row_counts, recv_cunts, displacements:\n");
    for (int i = 0; i < wsz; i++) printf("%d %d %d\n", row_counts[i], recv_counts[i], displ[i]);
    printf("\nnumber of iterations = %d\n", iterations);
    printf("element for checking = Mat[%d,%d]\n",row_peek, col_peek);
  }

  if(((row_peek > N) || (col_peek > N)) && rank == 0) {
    fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
    fprintf(stderr, "Arguments n and m must be smaller than %d\n", N);
    return 1;
  }

  matrix = (double *) malloc( row_counts[rank] * (N + 2) * sizeof(double) );
  matrix_new = (double *) malloc( row_counts[rank] * (N + 2) * sizeof(double) );
  memset( matrix, 0, row_counts[rank] * (N + 2) * sizeof(double) );
  memset( matrix_new, 0, row_counts[rank] * (N + 2) * sizeof(double) );

  // fill initial values  
  for (int i = 1; i < row_counts[rank]; i++) {
    for (int j = 1; j < N + 1; j++) { // not the first and last column!
      matrix[ ( i * ( N + 2 ) ) + j ] = 0.5;
    }
  }

  increment = 100.0 / ( N + 1 );  // set up borders

// the block goes from 0 to row_counts + 1 to accomodate one ghost row above and below
  int global_row;
  for (int i = 1; i < row_counts[rank]; i++) {
    global_row = displ[rank] + i;
    matrix[i * (N + 2)] = global_row * increment;
    matrix_new[i * (N + 2)] = global_row * increment;
  }
/*
if (rank == 0) {
  for (int i = 0; i <= row_counts[rank] + 1; i++) {
    for (int j = 0; j < N + 2; j++) {
      printf("%g ", matrix[i*(N+2) + j]);
    }
    printf("\n");
  }
  printf("-------------------------\n");
}*/

  // The last one fills the lower row
  if (rank == wsz - 1) {
    for (int i = 1; i < N + 1; ++i) {
      matrix[((row_counts[rank] - 1) * (N + 2)) + (N + 1 - i)] = i * increment;
      matrix_new[((row_counts[rank] - 1)* (N + 2)) + (N + 1 - i)] = i * increment;
    }
  }

if (wsz > 1) {
  if (rank == 0) {
    printf("RANK 0\n");
    printMatrix(matrix, row_counts[rank] - 1, N + 2); // print from the beginning
    for (int count = 1; count < wsz - 1; count++) {
        MPI_Recv(matrix + N + 2, (row_counts[count] - 2) * (N + 2), MPI_DOUBLE, count, count, 
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("RANK %d\n", count);
        printMatrix(matrix, row_counts[count] - 2, N + 2);
    } MPI_Recv(matrix + N + 2, (row_counts[wsz - 1] - 1) * (N + 2), MPI_DOUBLE, wsz - 1, wsz - 1, 
      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("RANK %d\n", wsz - 1);
      printMatrix(matrix, row_counts[wsz - 1] - 1, N + 2);     
  } else if (rank == wsz - 1) {
    MPI_Send(matrix + N + 2, (row_counts[rank] - 1) * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
  } else MPI_Send(matrix + N + 2, (row_counts[rank] - 2) * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
} else printMatrix(matrix, row_counts[rank] - 1, N + 2);
/*
  t_start = seconds();  // start algorithm
  for (int it = 0; it < iterations; it++) {
    evolve( matrix, matrix_new, N );
    // swap the pointers // very clever!
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  }
  t_end = seconds();
  
  printf( "\nelapsed time = %f seconds\n", t_end - t_start );
  printf( "\nmatrix[%d,%d] = %f\n", row_peek, col_peek, matrix[ ( row_peek + 1 ) * ( N + 2 ) + ( col_peek + 1 ) ] );
  save_gnuplot( matrix, N );*/
  free( matrix );
  free( matrix_new );

  MPI_Finalize();
  return 0;
}

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

void save_gnuplot( double *M, int N ) {

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

void printMatrix(const double* M, const int rows, const int cols) {
  for (int i = 1; i <= rows; i++ ) {
      for (int j = 0; j < cols; j++ ) {
          fprintf( stdout, "%.3g ", M[i*cols + j] );
      }
      fprintf( stdout, "\n");
  }
}
