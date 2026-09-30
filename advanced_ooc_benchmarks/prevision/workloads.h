/* Workload DAGs; setup, execution, import, and export live in adapter.c. */
#ifndef BENCH_PREVISION_WORKLOADS_H
#define BENCH_PREVISION_WORKLOADS_H

#include <stdint.h>
#include "node_interface.h"

/* PreVision borrows constants: epsilon must live until the DAG is executed. */
void prevision_gnmf(Array *X, Array **W, Array **H,
                   uint64_t iterations, double *epsilon);
Array *prevision_gram(Array *X);

#endif
