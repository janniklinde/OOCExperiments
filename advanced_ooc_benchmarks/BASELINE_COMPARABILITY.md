# Baseline correspondence

The NumPy and Dask scripts mirror the equations in each workload's vendored
`implementation.dml`. Their variable names follow those equations where practical.
Comparable logical work does not imply identical physical plans, I/O, memory
requirements, random streams, or bitwise-identical floating-point reductions.

| Workload | Correspondence and remaining distinctions |
| --- | --- |
| KMeans | `X`, `C`, `D`, `P`, `Y`: first-k initialization, fractional membership for tied minima, fixed Lloyd updates, and final assignment. Final ties select the last cluster, matching `rowIndexMin`. No separate empty-cluster fallback is added. |
| L2-SVM | `X`, `Y`, `w`, `Xw`, `g_old`, `s`, `Xd`, `step_sz`, `g`, `h`, `g_new`: squared hinge loss, label normalization, nonlinear CG, and Newton line search. Both inner and outer tolerance tests match DML. Python rejects invalid label sets rather than merely warning. |
| GNMF | `X`, `W`, `H`: identical positive modular initialization and H-then-W Lee–Seung updates. Dask persists W with native spilling; NumPy holds factors in memory. |
| MultiLogReg | `X`, `B`, `P`, `Grad`, `S`, `R`, `V`, `Q`, `HV`: same TRON constants, Hessian-vector products, trust boundary, acceptance, and initial/relative convergence tests. Nonpositive labels map to the baseline category as in DML. NaNs become zero. Baselines support the benchmark's `icpt=0` path, not DML's intercept/standardization options. |
| LMCG | `X`, `y`, `beta`, `r`, `p`, `q`, `norm_r2`: same zero initialization, implicit normal-equation products, regularization, and stopping test. Dask may share input reads between products; NumPy exposes the complete memmap to BLAS. |
| PCA | `X`, `sums`, `gram`, `center_correction`, `covariance`, `components`, `eigenvalues`, `scores`: same covariance identity and centered projection. Components correspond to decreasing eigenvalues. Signs, and bases in degenerate eigenspaces, may differ between eigensolvers. |
| MLP | `X`, `Y`, `W1`, `W2`, `b1`, `b2`, `dW1`, `dW2`, `lr`, `mu`: same affine/ReLU/dropout/affine/sigmoid path, unclipped log loss and chain-rule gradient, Nesterov update, and epoch schedules. He-normal initialization and dropout distributions match, but RNG streams do not; identical seeds do not imply identical models. NumPy uses memmaps and compact uint8 predicate masks, Dask uses boolean predicates and double-buffered Zarr parameters. |

All prepared feature matrices and arithmetic model/factor arrays use FP64.
Integer labels/indices and compact predicate masks are auxiliary representations,
not quantized feature matrices. NumPy/Dask use their native libraries and scheduling;
we do not impose SystemDS's block grid on them.

Dask is not guaranteed to remain out of core for every intermediate. In particular,
L2-SVM computes row-sized vectors into the client, and several workloads keep
small models or label arrays in the client. Extreme tall shapes can therefore
fail even when the input itself is chunked and spillable.

## Validation

Run the small numerical check with the benchmark Python environment:

```bash
python check_baseline_comparability.py --java java --systemds-jar /path/to/SystemDS.jar
```

It compares DML, NumPy, and Dask outputs for KMeans (including exact ties), LMCG,
GNMF, PCA (allowing eigenvector signs), L2-SVM (nonzero tolerance), and three-class
MultiLogReg. MLP receives NumPy/Dask smoke tests only because RNG streams differ.
These checks do not establish large-run performance or memory safety.

The PCA DML eigenvector permutation was corrected during this review. Existing
PCA outputs can have components that do not correspond to the reported dominant
eigenvalues; regenerate those correctness results with the updated implementation.
