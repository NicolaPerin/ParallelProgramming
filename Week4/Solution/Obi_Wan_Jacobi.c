#include "utility.h"

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int rank, wsz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsz);

  double t_start, t_end;// timing variables
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

  int* rows = (int *) malloc( wsz * sizeof(int) ); // N° of actual rows
  int* offset = (int *) malloc( wsz * sizeof(int) ); // For initialization

  initCounts(n_loc, wsz, rest, rows, offset);

  if (rank == 0) {
    printf("N (int): %d\nN (ext): %d\nn_loc: %d\nrest: %d\n\n", N, N + 2, n_loc, rest);
    for (int i = 0; i < wsz; i++) printf("%d %d\n", rows[i], offset[i]);
    printf("\nnumber of iterations = %d\n", iterations);
    printf("element for checking = Mat[%d,%d]\n",row_peek, col_peek);
  }

  if(((row_peek > N) || (col_peek > N)) && rank == 0) {
    fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
    fprintf(stderr, "Arguments n and m must be smaller than %d\n", N);
    return 1;
  }

  // allocation includes space for ghost rows
  matrix = (double*)malloc( (rows[rank] + 2) * (N + 2) * sizeof(double) );
  matrix_new = (double*)malloc( (rows[rank] + 2) * (N + 2) * sizeof(double) );

  initMatrix(rank, wsz, N, rows, offset, matrix);
  initMatrix(rank, wsz, N, rows, offset, matrix_new);

  // Exchange ghost rows
  int above = (rank == 0) ? MPI_PROC_NULL : rank - 1; // First process doesn't send its upper row to anybody
  int below = (rank == wsz - 1) ? MPI_PROC_NULL : rank + 1; // Last process doesn't sent its lower row to anybody

  t_start = seconds();  // start algorithm
  for (int it = 0; it < iterations; it++) {
    // Exchange upper row
    MPI_Sendrecv(matrix + N + 2, N + 2, // Send the first (actual) row (the second)
                 MPI_DOUBLE, above, 0, // to the process above
                 matrix + (rows[rank] + 1 ) * (N + 2), N + 2, // Receive it and put it in the lower ghost row (last row)
                 MPI_DOUBLE, below, 0, // from the process below (same tag)
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Exchange lower row
    MPI_Sendrecv(matrix + rows[rank] * (N + 2), N + 2, // Send the last (actual) row (the second to last)
                 MPI_DOUBLE, below, 0, // to the process below with tag = rank + wsz + 1
                 matrix, N + 2, // Receive the row and put it in the upper ghost row (the first row)
                 MPI_DOUBLE, above, 0, // from the process above (same tag)
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    evolve( matrix, matrix_new, rows, rank, N );
    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  }
  t_end = seconds();
  
  if (rank == 0) printf( "\nelapsed time = %f seconds\n", t_end - t_start );
  //printf( "\nmatrix[%d,%d] = %f\n", row_peek, col_peek, matrix[ ( row_peek + 1 ) * ( N + 2 ) + ( col_peek + 1 ) ] );
  //save_gnuplot( matrix, N );

  printCalls(wsz, rank, N, rows, matrix); // Send to process zero to be printed

  free( matrix );
  free( matrix_new );

  MPI_Finalize();
  return 0;
}
