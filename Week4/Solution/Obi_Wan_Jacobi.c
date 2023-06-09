#include "utility.h"

int main(int argc, char* argv[]) {

  MPI_Init(&argc, &argv);
  int rank, wsz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &wsz);

  double t_start, t_end, t_elapsed, t_evolve = 0.0; // timing variables
  double *restrict matrix, *restrict matrix_new;
  size_t N = 0;
  int iterations = 0, print = 0;

  // check on input parameters
  if (argc != 4) {
    if (rank == 0) {
      red(); printf("\nWrong number of arguments.\nUsage: mpirun -np <number of processes> jacobi.x <dim> <iterations> <print>\n"); reset();
    }
    return 1;
  }

  N = atoi(argv[1]); // 9
  iterations = atoi(argv[2]);
  print = atoi(argv[3]); // 0 doesn't print but writes to file

  // Check on grid size vs number of processes
  if (wsz > N) {
    if (rank == 0) {
      red(); printf("The number of MPI processes (%d) is greater than the number of rows of the matrix (%d)\n", wsz, N); reset();
    }
    return 1;
  }

  size_t n_loc = N / wsz;
  size_t rest = N % wsz;
  int* rows = (int *) malloc( wsz * sizeof(int) ); // N° of actual rows
  int* offset = (int *) malloc( wsz * sizeof(int) ); // For initialization

  initCounts(n_loc, wsz, rest, rows, offset);

  if (print && rank == 0) {
    printf("N (int): %d\nN (ext): %d\nn_loc: %d\nrest: %d\n", N, N + 2, n_loc, rest);
    printf("rows, offset\n");
    for (int i = 0; i < wsz; i++) printf("%d %d\n", rows[i], offset[i]);
    printf("number of iterations = %d\n", iterations);
  }

  // Allocation includes space for ghost rows
  matrix = (double*) malloc ((rows[rank] + 2) * (N + 2) * sizeof(double) );
  matrix_new = (double*) malloc ((rows[rank] + 2) * (N + 2) * sizeof(double) );

  initMatrix(rank, wsz, N, rows, offset, matrix); // Initial conditions
  initMatrix(rank, wsz, N, rows, offset, matrix_new);

  t_start = MPI_Wtime();

  Jacobi(matrix, matrix_new, rows, rank, wsz, N, iterations, &t_evolve); // Simulation
  if (rank == 0) { green(); printf("Simulation ok!\n"); reset(); }

  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();
  t_elapsed = t_end - t_start;

  t_start = MPI_Wtime();
  if (!print) writeCalls(wsz, rank, N, rows, matrix); // Write to file in order
  MPI_Barrier(MPI_COMM_WORLD);
  t_end = MPI_Wtime();

  if (!print && rank == 0) { green(); printf("Writing ok!\n"); reset(); }

  if (rank == 0) {
    printf("\nComputation time: %f s\n", t_evolve);
    printf("Communication time: %f seconds\n", t_elapsed - t_evolve);
    if (!print) printf("Writing time: %f s\n", t_end - t_start);
#ifdef ACC
    FILE *fp = fopen("ACC_scalability.txt", "a");
#else
    FILE *fp = fopen("MPI_scalability.txt", "a");
#endif
    fprintf(fp, "%d %d %d %g %g %g\n", N + 2, wsz, iterations, t_evolve, t_elapsed - t_evolve);
    fclose(fp);
  }

  if (print && N < 50) printCalls(wsz, rank, N, rows, matrix); // Send to process zero to be printed
  if (print && rank == 0) printf("\n-------------------------------------\n");
  if (print && N < 50) printCalls(wsz, rank, N, rows, matrix_new); // Send to process zero to be printed

  free(matrix);
  free(matrix_new);

  MPI_Finalize();
  return 0;
}
