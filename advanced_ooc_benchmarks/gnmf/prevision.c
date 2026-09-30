/* FP64 Lee-Seung multiplicative updates using PreVision's lazy array API.
 * Matches implementation.dml: update H first, then W, with epsilon in both
 * denominators. The shared adapter prepares inputs and materializes outputs. */
#include "../prevision/workloads.h"
#include "exec_interface.h"

void prevision_gnmf(Array *X, Array **W_out, Array **H_out,
                   uint64_t iterations, double *epsilon) {
    Array *W = *W_out, *H = *H_out;
    /* transpose() borrows this array until the lazy DAG is executed. */
    static uint32_t order[2] = {1, 0};
    for (uint64_t i = 0; i < iterations; i++) {
        Array *Wt = transpose(W, order);
        Array *Hden = matmul(matmul(Wt, W), H);
        Array *Hden_eps = elemwise_op_constant(Hden, epsilon, TILESTORE_FLOAT64,
                                               RHS, OP_ADD_CONSTANT);
        Array *Hnext = elemwise_op(H, elemwise_op(matmul(Wt, X), Hden_eps,
                                                  OP_DIV), OP_PRODUCT);
        Array *Ht = transpose(Hnext, order);
        Array *Wden = matmul(W, matmul(Hnext, Ht));
        Array *Wden_eps = elemwise_op_constant(Wden, epsilon, TILESTORE_FLOAT64,
                                               RHS, OP_ADD_CONSTANT);
        W = elemwise_op(W, elemwise_op(matmul(X, Ht), Wden_eps, OP_DIV),
                        OP_PRODUCT);
        H = Hnext;
    }
    *W_out = W;
    *H_out = H;
}
