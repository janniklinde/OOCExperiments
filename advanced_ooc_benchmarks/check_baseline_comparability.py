#!/usr/bin/env python3
"""Small DML/NumPy/Dask numerical checks; not a performance benchmark.

Pass --java and --systemds-jar for the build under test. Temporary validation
artifacts and logs are retained and their directory is printed for inspection.
MLP is smoke-tested only: framework-specific RNG streams prevent exact outputs.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--java', default='java')
    parser.add_argument('--systemds-jar', type=Path, required=True)
    args = parser.parse_args()
    suite = Path(__file__).resolve().parent
    root = Path(tempfile.mkdtemp(prefix='baseline-check-'))
    print('Validation artifacts:', root, flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    rng = np.random.default_rng(421)
    cases = {
        'kmeans': (['--clusters', '2', '--iterations', '3'],
            '[C,Y,inertia]=impl::m_kmeans_fixed(X=X,k=2,iterations=3);',
            [('C', 'centers'), ('Y', 'labels')]),
        'lmcg': (['--iterations', '3'],
            '[beta,residual_norm]=impl::m_lmcg_fixed(X=X,y=Y,reg=1e-7,iterations=3);', [('beta', 'beta')]),
        'gnmf': (['--rank', '3', '--iterations', '3'],
            '[W,H]=impl::m_gnmf(X=X,rank=3,iterations=3,seed=23);', [('W', 'W'), ('H', 'H')]),
        'pca': (['--components', '3'],
            '[scores,components,eigenvalues]=impl::m_pca_covariance(X=X,k=3);',
            [('components', 'components'), ('eigenvalues', 'eigenvalues')]),
        'l2svm': (['--iterations', '3', '--tolerance', '.01'],
            'w=impl::m_l2svm(X=X,Y=Y,intercept=FALSE,epsilon=.01,reg=1,maxIterations=3,maxii=20);',
            [('w', 'model')]),
        'multilogreg': (['--iterations', '3', '--inner-iterations', '4', '--tolerance', '0'],
            'B=impl::m_multiLogReg(X=X,Y=Y,icpt=0,tol=0,reg=1,maxi=3,maxii=4,verbose=FALSE);', [('B', 'B')]),
        'mlp': (['--hidden-size', '8', '--epochs', '1', '--batch-size', '12'], '', []),
    }
    for algorithm, (options, expression, pairs) in cases.items():
        data = root / algorithm
        data.mkdir()
        X = rng.normal(size=(48, 5)) * [3, .2, 2, .5, 1]
        if algorithm == 'gnmf':
            X = np.abs(X) + .1
        if algorithm == 'kmeans':
            X[:2] = 0  # identical centroids exercise update and final-label ties
        y = np.where(X[:, 0] > 0, 1., -1.).reshape(-1, 1)
        Y = (y > 0).astype(float)
        if algorithm == 'multilogreg':
            Y = (np.arange(48) % 3 + 1).reshape(-1, 1).astype(float)
        for name, array in [('X', X), ('binary_y', y), ('nn_y', Y)]:
            array.tofile(data / (name + '.f64'))
        metadata = dict(rows=48, cols=5, classes=3 if algorithm == 'multilogreg' else 2)
        for name in ('metadata.json', 'X.f64.json'):
            (data / name).write_text(json.dumps(metadata))
        store = zarr.open_array(str(data / 'zarr' / 'X.zarr'), mode='w',
                               shape=X.shape, chunks=(12, 5), dtype='f8')
        store[:] = X
        np.savetxt(data / 'X.csv', X, delimiter=',')
        np.savetxt(data / 'Y.csv', Y if algorithm == 'multilogreg' else y, delimiter=',')
        for backend, filename in [('numpy', 'numpy.py'), ('dask', 'dask_array.py')]:
            command = [sys.executable, str(suite / algorithm / filename), str(data),
                       *options, '--output', str(data / (backend + '.json'))]
            if backend == 'dask':
                command += ['--threads', '2', '--workers', '1', '--memory-limit', '1GiB',
                            '--temporary-directory', str(data / 'tmp')]
                if algorithm == 'mlp':
                    command += ['--hidden-chunk', '4']
            run(command, suite, env, data / (backend + '.log'))
        if not pairs:
            print(algorithm, 'NumPy/Dask smoke PASS (different RNGs)', flush=True)
            continue
        script = (f'source("{algorithm}/implementation.dml") as impl;'
                  f'X=read("{data}/X.csv",format="csv",rows=48,cols=5);'
                  f'Y=read("{data}/Y.csv",format="csv",rows=48,cols=1);' + expression)
        for variable, suffix in pairs:
            script += f'write({variable},"{data}/{suffix}.csv",format="csv");'
        opens = [f'--add-opens=java.base/{package}=ALL-UNNAMED' for package in (
            'java.nio', 'java.io', 'java.util', 'java.lang', 'java.lang.ref',
            'java.lang.invoke', 'java.util.concurrent', 'sun.nio.ch')]
        run([args.java, '-Xmx1g', '--add-modules=jdk.incubator.vector', *opens, '-jar',
             str(args.systemds_jar.resolve()), '-s', script],
            suite, env, data / 'dml.log')
        for _, suffix in pairs:
            expected = np.loadtxt(data / (suffix + '.csv'), delimiter=',', ndmin=2)
            for backend in ('numpy', 'dask'):
                extension = '.zarr' if backend == 'dask' and algorithm == 'gnmf' and suffix == 'W' else '.npy'
                path = data / (backend + '-' + suffix + extension)
                actual = np.asarray(zarr.open(str(path))) if extension == '.zarr' else np.load(path)
                actual = actual.reshape(expected.shape)
                if algorithm == 'pca' and suffix == 'components':
                    actual *= np.sign(np.sum(actual * expected, axis=0))
                np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9,
                                           err_msg=f'{algorithm} {backend} {suffix}')
        print(algorithm, 'DML/NumPy/Dask numerical PASS', flush=True)


def run(command, cwd, env, log):
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    log.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed; see {log}\n{result.stdout[-2500:]}')


if __name__ == '__main__':
    main()
