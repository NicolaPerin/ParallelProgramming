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
  int* ghost_counts = (int *) malloc( wsz * sizeof(int) );
  int* recv_counts = (int *) malloc( wsz * sizeof(int) );
  int* displ = (int *) malloc( wsz * sizeof(int) );

  // If rank < rest, the process gets (N + 2) / wsz + 3 rows, otherwise, + 2
  for (int i = 0; i < wsz; i++) { ghost_counts[i] = (i < rest) ? n_loc + 3 : n_loc + 2; }
  for (int i = 0; i < wsz; i++) recv_counts[i] = ghost_counts[i] - 2;
  ghost_counts[0] -= 1; ghost_counts[wsz - 1] -= 1; // first and last processes only have 1 ghost rows
  displ[0] = 0;
  for (int i = 1; i <= wsz; i++) displ[i] = displ[i-1] + recv_counts[i-1];

  if (rank == 0) {
    printf("matrix size: (int) %d (ext) %d\n", N, N + 2);
    printf("ghost_counts, recv_counts, displacements:\n");
    for (int i = 0; i < wsz; i++) printf("%d %d %d\n", ghost_counts[i], recv_counts[i], displ[i]);
    printf("\nnumber of iterations = %d\n", iterations);
    printf("element for checking = Mat[%d,%d]\n",row_peek, col_peek);
  }

  if(((row_peek > N) || (col_peek > N)) && rank == 0) {
    fprintf(stderr, "Cannot Peek a matrix element outside of the matrix dimension\n");
    fprintf(stderr, "Arguments n and m must be smaller than %d\n", N);
    return 1;
  }

  matrix = (double *) malloc( ghost_counts[rank] * (N + 2) * sizeof(double) );
  matrix_new = (double *) malloc( ghost_counts[rank] * (N + 2) * sizeof(double) );
  memset( matrix, 0, ghost_counts[rank] * (N + 2) * sizeof(double) );
  memset( matrix_new, 0, ghost_counts[rank] * (N + 2) * sizeof(double) );

  initMatrix(rank, wsz, N, ghost_counts, displ, matrix);
  initMatrix(rank, wsz, N, ghost_counts, displ, matrix_new);

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
  save_gnuplot( matrix, N );
*/

  printCalls(wsz, rank, N, recv_counts, matrix); // Send to process zero to be printed

  free( matrix );
  free( matrix_new );

  MPI_Finalize();
  return 0;
}
