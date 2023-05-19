#include "utility.h"

#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"

void reset() { printf(RESET); }
void red() { printf(RED); }
void green() { printf(GREEN); }
void yellow() { printf(YELLOW); }
void blue() { printf(BLUE); }

void evolve(double* matrix, double* matrix_new, const int* rows, const int rank, const int N, const int nelems) {
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

void Jacobi(double* matrix, double* matrix_new, const int* rows, const int rank, const int wsz,
            const int N, const int iterations, double* t_evolve) {

  // Who are my neighbours
  int above = (rank == 0) ? MPI_PROC_NULL : rank - 1; // First process doesn't send its upper row to anybody
  int below = (rank == wsz - 1) ? MPI_PROC_NULL : rank + 1; // Last process doesn't sent its lower row to anybody

  int nelems = (rows[rank] + 2) * (N + 2); // Number of elements of the grid, including ghost rows

#ifdef ACC // OpenACC stuff

  const acc_device_t devtype = acc_get_device_type(); // Device type (e.g. Tesla)
  const int num_devs = acc_get_num_devices(devtype); // Number of devices per node
  acc_set_device_num(rank % num_devs, devtype); // To run on multiple nodes
  acc_init(devtype);

  #pragma acc enter data copyin(matrix[:nelems], matrix_new[:nelems])
  int start = (rows[rank] + 1) * (N + 2); // Why do I have to do this
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

  // The last one fills the last row
  if (rank == wsz - 1) {
    for (int i = 1; i <= N + 1; ++i) matrix[(rows[rank] + 1) * (N + 2) + (N + 1 - i)] = i * increment;
  }
}

void exchangeRows(double* matrix, const int* rows, const int N, const int rank, const int above, const int below) {
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

void hsvToRgb(float h, float s, float v, int *r, int *g, int *b) {
    float f = h / 60 - floor(h / 60);
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    if (h < 60) {
        *r = (int) (v * 255); *g = (int) (t * 255); *b = (int) (p * 255);
    } else if (h < 120) {
        *r = (int) (q * 255); *g = (int) (v * 255); *b = (int) (p * 255);
    } else if (h < 180) {
        *r = (int) (p * 255); *g = (int) (v * 255); *b = (int) (t * 255);
    } else if (h < 240) {
        *r = (int) (p * 255); *g = (int) (q * 255); *b = (int) (v * 255);
    } else if (h < 300) {
        *r = (int) (t * 255); *g = (int) (p * 255); *b = (int) (v * 255);
    } else {
        *r = (int) (v * 255); *g = (int) (p * 255); *b = (int) (q * 255);
    }
}

void getColor(int value, int min, int max, int *r, int *g, int *b) {
    float ratio = (float)(value - min) / (max - min);
    float hue = 240 - ratio * 240;
    hsvToRgb(hue, 1.0, 1.0, r, g, b);
}

void printMatrix(const double* M, const int rows, const int cols, const int rank, const int wsz) {
  int r, g, b;
  if (!rank) {
    for (int j = 0; j < cols; j++ ) {
      getColor(M[j], 0, 100, &r, &g, &b);
      printf("\033[38;2;%d;%d;%dm%.3g%s ", r, g, b, M[j], RESET);
    }
  } else for (int j = 0; j < cols; j++ ) printf("%.3g ", M[j]);
  reset();
  printf("\n");
  for (int i = 1; i < rows - 1; i++ ) {
      for (int j = 0; j < cols; j++ ) {
          getColor(M[i*cols + j], 0, 100, &r, &g, &b);
          printf("\033[38;2;%d;%d;%dm%.3g%s ", r, g, b, M[i*cols + j], RESET);
      }
      printf("\n");
  }
  if (rank == wsz - 1) {
    for (int j = 0; j < cols; j++ ) {
      getColor(M[(rows - 1)*cols + j], 0, 100, &r, &g, &b);
      printf("\033[38;2;%d;%d;%dm%.3g%s ", r, g, b, M[(rows - 1)*cols + j], RESET);
    }
  } else for (int j = 0; j < cols; j++ ) printf("%.3g ", M[(rows - 1)*cols + j]);
  reset();
  printf("\n");
}

// print with ghost rows
void printCalls(const int wsz, const int rank, const int N, const int* rows, double* matrix) {
  // Print in order
  if (rank == 0) {
      yellow(); printf("Rank %d:\n", rank); reset();
      printMatrix(matrix, rows[rank] + 2, N + 2, rank, wsz);
      for (int count = 1; count < wsz; count++) {
          MPI_Recv(matrix, (rows[count] + 2) * (N + 2), MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          yellow(); printf("Rank %d:\n", count); reset();
          printMatrix(matrix, rows[count] + 2, N + 2, count, wsz);
      }
  } else MPI_Send(matrix, (rows[rank] + 2) * (N + 2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
}

void save_gnuplot(const double* M, const size_t N, int *rows, const int rank, const int wsz) {
    size_t i, j;
    const double h = 0.1;
    int row_sum;
    MPI_File file;
    MPI_Offset start_write;

    if (rank == 0) rows[rank] += 1;
    MPI_Barrier( MPI_COMM_WORLD );

    MPI_Scan(rows + rank, &row_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    start_write = (row_sum - rows[rank]) * (N + 2) * 3 * sizeof(double);

    MPI_Barrier( MPI_COMM_WORLD );
    if (rank == 0) rows[rank] -= 1;

    double data[3];
    int* offset = (int*) malloc (wsz * sizeof(int));
    offset[0] = 0;
    for (i = 1; i < wsz; i++) offset[i] = offset[i-1] + rows[i-1];

    MPI_File_open(MPI_COMM_WORLD, "solution.dat", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);

    int start = (rank == 0) ? 0 : 1; // First process writes also its upper ghost row
    int end = (rank == wsz - 1) ? rows[rank] + 1 : rows[rank]; // Last process writes also its lower ghost row

    printf("rank %d start %d end %d\n", rank, start, end);

    for (i = start; i <= end; ++i) {
        printf("OK %d\n", i + offset[rank]);
        for (j = 0; j < N + 2; ++j) {
            data[0] = h * j;
            data[1] = -h * (i + offset[rank]);
            data[2] = M[(i * (N + 2)) + j];
            MPI_File_write_at_all(file, start_write, data, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
            start_write += 3 * sizeof(double);
        }
    }

    MPI_File_close(&file);
}
