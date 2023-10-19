#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "likwid-stuff.h"

const char *dgemm_desc = "Blocked dgemm, OpenMP-enabled";

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm_blocked(int n, int block_size, double *A, double *B, double *C)
{
// insert your code here: implementation of blocked matrix multiply with copy optimization and OpenMP parallelism enabled

// be sure to include LIKWID_MARKER_START(MY_MARKER_REGION_NAME) inside the block of parallel code,
// but before your matrix multiply code, and then include LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME)
// after the matrix multiply code but before the end of the parallel code block.
#pragma omp parallel for
   {
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
      double A_block[block_size * block_size];
      double B_block[block_size * block_size];
      double C_block[block_size * block_size]; // Adding the block for C
      int size = block_size * sizeof(double);

      for (int si = 0; si < n; si += block_size)
      {
         for (int sj = 0; sj < n; sj += block_size)
         {
            // Read/copy block C[si,sj] into cache
            for (int x = 0; x < block_size; ++x)
            {
               memcpy(&C_block[x * block_size], &C[si + (x + sj) * n], size);
            }

            for (int sk = 0; sk < n; sk += block_size)
            {
               // Copy blocks of A and B into local storage using memcpy for column-major storage
               for (int i = 0; i < block_size; ++i)
               {
                  memcpy(&A_block[i * block_size], &A[si + (sk + i) * n], size);
                  memcpy(&B_block[i * block_size], &B[sk + (sj + i) * n], size);
               }

               // Multiply the individual blocks using local storage
               for (int i = 0; i < block_size; ++i)
               {
                  for (int j = 0; j < block_size; ++j)
                  {
                     int index = i + j * block_size;
                     for (int k = 0; k < block_size; ++k)
                     {
                        C_block[index] += A_block[i + k * block_size] * B_block[k + j * block_size];
                     }
                  }
               }
            }

            // Write/copy block C[si,sj] back to memory
            for (int x = 0; x < block_size; ++x)
            {
               memcpy(&C[si + (x + sj) * n], &C_block[x * block_size], size);
            }
         }
      }
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
   }
}
