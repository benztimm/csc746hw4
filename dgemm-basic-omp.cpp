#include <iostream>
#include <omp.h>
#include "likwid-stuff.h"

const char *dgemm_desc = "Basic implementation, OpenMP-enabled, three-loop dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double *A, double *B, double *C)
{
#pragma omp parallel
   {
#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(MY_MARKER_REGION_NAME);
#endif

#pragma omp for collapse(2)
      for (int i = 0; i < n; ++i)
      {
         for (int j = 0; j < n; ++j)
         {
            double cij = C[i + j*n];
            for (int k = 0; k < n; ++k)
            {
               cij += A[i + k * n] * B[k + j * n]; // A and B are in column-major format
            }
            C[i + j*n] = cij;
         }
      }

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(MY_MARKER_REGION_NAME);
#endif
   }
}