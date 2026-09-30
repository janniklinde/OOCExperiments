/* FP64 Gram matrix using PreVision's lazy array API. */
#include "../prevision/workloads.h"

Array *prevision_gram(Array *X) {
    /* transpose() borrows this array until the lazy DAG is executed. */
    static uint32_t order[2] = {1, 0};
    return matmul(transpose(X, order), X);
}
