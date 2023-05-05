#include "utility.h"

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"

void reset() { printf(RESET); }
void red() { printf(RED); }
void green() { printf(GREEN); }
void yellow() { printf(YELLOW); }

void evolve(double* matrix, double* matrix_new, const size_t* rows, const size_t rank, const size_t N, const size_t nelems) {
#ifdef ACC
  #pragma acc parallel loop collapse(2) present(matrix[:nelems], matrix_new[:nelems])
#endif
  for (int i = 1 ; i <= rows[rank]; i++)
    for (int j = 1; j <= N; j++)
      matrix_new[ ( i * ( N + 2 ) ) + j ] = ( 0.25 ) *
        ( matrix[ ( ( i - 1 ) * ( N + 2 ) ) + j ] +
          matrix[ ( i * ( N + 2 ) ) + ( j + 1 ) ] +
          matrix[ ( ( i + 1 ) * ( N + 2 ) ) + j ] +
          matrix[ ( i * ( N + 2 ) ) + ( j - 1 ) ] );
}

void Jacobi(double* matrix, double* matrix_new, const size_t* rows, const size_t rank, const size_t wsz,
            const size_t N, const size_t iterations, double* t_evolve) {

  // Who are my neighbours
  size_t above = (rank == 0) ? MPI_PROC_NULL : rank - 1; // First process doesn't send its upper row to anybody
  size_t below = (rank == wsz - 1) ? MPI_PROC_NULL : rank + 1; // Last process doesn't sent its lower row to anybody

  size_t nelems = (rows[rank] + 2) * (N + 2); // Number of elements of the grid, including ghost rows

#ifdef ACC // OpenACC stuff

  const acc_device_t devtype = acc_get_device_type(); // Device type (e.g. Tesla)
  const int num_devs = acc_get_num_devices(devtype); // Number of devices per node
  acc_set_device_num(rank % num_devs, devtype); // To run on multiple nodes
  acc_init(devtype);

  #pragma acc enter data copyin(matrix[:nelems], matrix_new[:nelems])
  size_t start = (rows[rank] + 1) * (N + 2); // Why do I have to do this
#endif

  double t_evolve_start, t_evolve_end;

  for (int it = 0; it < iterations; it++) {
#ifdef ACC
    if (it % 2 == 0) {
    #pragma acc update host(matrix_new[N + 2 : N + 2], matrix_new[start - (N + 2) : N + 2])
    } else {
    #pragma acc update host(matrix[N + 2 : N + 2], matrix[start - (N + 2) : N + 2])
    }
#endif

    if (it % 2 == 0) exchangeRows(matrix_new, rows, N, rank, above, below); // Exchange ghost rows
    else exchangeRows(matrix, rows, N, rank, above, below);
#ifdef ACC // Update the ghost rows on the device; parallelize the loop
    if (it % 2 == 0) {
    #pragma acc update device(matrix_new[0 : N + 2], matrix_new[start : N + 2])
    } else {
    #pragma acc update device(matrix[0 : N + 2], matrix[start : N + 2])
    }
#endif

    t_evolve_start = MPI_Wtime(); // Start computation timer
    if (it % 2 == 0) evolve(matrix_new, matrix, rows, rank, N, nelems); // Actual grid update
    else evolve(matrix, matrix_new, rows, rank, N, nelems);
    t_evolve_end = MPI_Wtime();
    *t_evolve += t_evolve_end - t_evolve_start; // Accumulate
  }

#ifdef ACC // Copy matrix back to host
  #pragma acc exit data copyout(matrix[:nelems], matrix_new[:nelems])
#endif
}

void initCounts(const size_t n_loc, const size_t wsz, const size_t rest, size_t* rows, size_t* offset) {
  for (int i = 0; i < wsz; i++) rows[i] = (i < rest) ? n_loc + 1 : n_loc; // first processes get the remainder
  offset[0] = 0;
  for (int i = 1; i < wsz; i++) offset[i] = offset[i-1] + rows[i-1];
}

void initMatrix(const size_t rank, const size_t wsz, const size_t N,
                const size_t* rows, const size_t* offset, double* matrix) {
  memset( matrix, 0, (rows[rank] + 2) * (N + 2) * sizeof(double) ); // set to zero to check for bugs later
  // fill initial values
  for (int i = 1; i <= rows[rank]; i++) // not the first row
    for (int j = 1; j < N + 1; j++) // not the first and last column either
      matrix[ ( i * ( N + 2 ) ) + j ] = 0.5;

  const double increment = 100.0 / ( N + 1 );  // set up borders

  for (int i = 1; i <= rows[rank]; i++) matrix[i * (N + 2)] = (i + offset[rank]) * increment;

  // The last one fills the last row
  if (rank == wsz - 1) {
    for (int i = 1; i <= N + 1; ++i) matrix[(rows[rank] + 1) * (N + 2) + (N + 1 - i)] = i * increment;
  }
}

void exchangeRows(double* matrix, const size_t* rows, const size_t N, const size_t rank, const size_t above, const size_t below) {
    // Exchange upper row
    MPI_Sendrecv(matrix + N + 2, N + 2, // Send the first (actual) row (the second)
                 MPI_DOUBLE, above, 0, // to the process above
                 matrix + (rows[rank] + 1 ) * (N + 2), N + 2, // Receive it and put it in the lower ghost row (last row)
                 MPI_DOUBLE, below, 0, // from the process below
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Exchange lower row
    MPI_Sendrecv(matrix + rows[rank] * (N + 2), N + 2, // Send the last (actual) row (the second to last)
                 MPI_DOUBLE, below, 0, // to the process below
                 matrix, N + 2, // Receive the row and put it in the upper ghost row (the first row)
                 MPI_DOUBLE, above, 0, // from the process above
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void printMatrix(const double* M, const size_t rows, const size_t cols) {
  for (int i = 0; i < rows; i++ ) {
      for (int j = 0; j < cols; j++ ) {
          fprintf( stdout, "%.3g ", M[i*cols + j] );
      }
      fprintf( stdout, "\n");
  }
}

// with ghost rows
void printCalls(const size_t wsz, const size_t rank, const size_t N, const size_t* rows, double* matrix) {
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

void save_gnuplot(double *M, size_t N, size_t *rows, size_t rank, size_t wsz) {
    size_t i, j;
    const double h = 0.1;
    MPI_File file;
    MPI_Offset offset;
    size_t displacement[wsz];
    size_t sum = 0;
    size_t data_size = rows[rank] * (N + 2) * 3;
    double data[data_size];

    for (i = 0; i < wsz; ++i) {
        displacement[i] = sum;
        sum += rows[i];
    }

    MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    // The first process has to write also its upper ghost rows and the last has to write its lower ghost row
    // The processes in the middle do not write their ghost rows
    size_t n_rows = (rank == 0 || rank == wsz - 1) ? rows[rank] + 1 : rows[rank];
    // The first process writes from the beginning, the others from the second row
    for (i = (rank == 0 ? 0 : 1); i < n_rows; i++) {
        for (j = 0; j < N + 2; j++) {
            data[(i * (N + 2) + j) * 3] = h * j;
            data[(i * (N + 2) + j) * 3 + 1] = -h * (i + displacement[rank]);
            data[(i * (N + 2) + j) * 3 + 2] = M[(i * (N + 2)) + j];
        }
    }

    offset = displacement[rank] * (N + 2) * sizeof(data[0]) * 3;
    MPI_File_write_at_all(file, offset, data, data_size, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_close(&file);
}
