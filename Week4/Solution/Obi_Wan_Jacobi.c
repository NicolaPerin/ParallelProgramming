#include "utility.h"

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int rank, wsz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsz);

  double t_start, t_end, t_elapsed, t_evolve_start, t_evolve_end, t_evolve = 0.0; // timing variables
  double *matrix, *matrix_new, *tmp_matrix; // initialize matrix + swap pointer
  int N = 0, iterations = 0, print = 0;

  // check on input parameters
  if(argc != 4) {
    fprintf(stderr,"\nwrong number of arguments. Usage: ./a.out dim it n m\n");
    return 1;
  }

  N = atoi(argv[1]); // 9
  iterations = atoi(argv[2]);
  print = atoi(argv[3]);

  int n_loc = N / wsz;
  int rest = N % wsz;
  int* rows = (int *) malloc( wsz * sizeof(int) ); // N° of actual rows
  int* offset = (int *) malloc( wsz * sizeof(int) ); // For initialization

  initCounts(n_loc, wsz, rest, rows, offset);

  if (rank == 0) {
    printf("N (int): %d\nN (ext): %d\nn_loc: %d\nrest: %d\n", N, N + 2, n_loc, rest);
    printf("rows, offset\n");
    for (int i = 0; i < wsz; i++) printf("%d %d\n", rows[i], offset[i]);
    printf("\nnumber of iterations = %d\n", iterations);
    // printf("element for checking = Mat[%d,%d]\n",row_peek, col_peek);
  }

  // allocation includes space for ghost rows
  matrix = (double*)malloc( (rows[rank] + 2) * (N + 2) * sizeof(double) );
  matrix_new = (double*)malloc( (rows[rank] + 2) * (N + 2) * sizeof(double) );

  initMatrix(rank, wsz, N, rows, offset, matrix);
  initMatrix(rank, wsz, N, rows, offset, matrix_new);

  // Exchange ghost rows
  int above = (rank == 0) ? MPI_PROC_NULL : rank - 1; // First process doesn't send its upper row to anybody
  int below = (rank == wsz - 1) ? MPI_PROC_NULL : rank + 1; // Last process doesn't sent its lower row to anybody

  t_start = MPI_Wtime(); // start algorithm

  for (int it = 0; it < iterations; it++) {
    exchangeRows(matrix, rows, N, rank, above, below);
    t_evolve_start = MPI_Wtime();
    evolve( matrix, matrix_new, rows, rank, N );
    t_evolve_end = MPI_Wtime();
    t_evolve += t_evolve_end - t_evolve_start;
    // swap the pointers
    tmp_matrix = matrix;
    matrix = matrix_new;
    matrix_new = tmp_matrix;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  t_elapsed = t_end - t_start;

  if (rank == 0) printf("\nEvolve time = %f seconds\n", t_evolve);
  if (rank == 0) printf("Total elapsed time = %f seconds\n", t_elapsed);

  //save_gnuplot( matrix, N );

  if (print && N < 50) printCalls(wsz, rank, N, rows, matrix_new); // Send to process zero to be printed

  free( matrix );
  free( matrix_new );

  MPI_Finalize();
  return 0;
}
