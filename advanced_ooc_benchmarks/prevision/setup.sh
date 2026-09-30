#!/usr/bin/env bash
# Build the published PreVision engine and this suite's small FP64 adapter.
set -euo pipefail

prefix="${1:?usage: setup.sh PREFIX [EXISTING_SOURCE]}"
source_dir="${2:-$prefix/source}"
commit=e4c4e96cd2e884b6f0d97e6b3410623fa4d13f59
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

command -v gcc >/dev/null
command -v make >/dev/null
command -v git >/dev/null
mkdir -p "$prefix/bin"
if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none https://github.com/snu-dbs/prevision.git "$source_dir"
fi
if [[ "$(git -C "$source_dir" rev-parse HEAD)" != "$commit" ]]; then
  git -C "$source_dir" fetch --depth 1 origin "$commit"
  git -C "$source_dir" checkout --detach "$commit"
fi

# PreVision's Makefiles use these standard compiler search paths. On hosts with
# OpenBLAS in a nonstandard prefix, set PREVISION_BLAS_INCLUDE and LIB before setup.
if [[ -n "${PREVISION_BLAS_INCLUDE:-}" ]]; then
  export CPATH="${PREVISION_BLAS_INCLUDE}${CPATH:+:$CPATH}"
fi
if [[ -n "${PREVISION_BLAS_LIB:-}" ]]; then
  export LIBRARY_PATH="${PREVISION_BLAS_LIB}${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi

make -C "$source_dir/tilestore" libtilestore.a -j4
make -C "$source_dir/buffertile" libbf.a -j4
make -C "$source_dir/tilechunk" libchunk.a -j4
make -C "$source_dir/linear_algebra_module" liblam.a -j4
make -C "$source_dir/lam_executor" libexec.a -j4

blas_rpath=()
if [[ -n "${PREVISION_BLAS_LIB:-}" ]]; then
  blas_rpath=("-Wl,-rpath,$PREVISION_BLAS_LIB")
fi
gcc -std=gnu11 -O2 \
  -I"$source_dir/tilestore/include" -I"$source_dir/buffertile/include" \
  -I"$source_dir/tilechunk/include" -I"$source_dir/linear_algebra_module/include" \
  -I"$source_dir/lam_executor/include" \
  "$script_dir/adapter.c" "$script_dir/../gnmf/prevision.c" \
  "$script_dir/../gram/prevision.c" \
  -L"$source_dir/lam_executor/lib" -L"$source_dir/linear_algebra_module/lib" \
  -L"$source_dir/tilechunk/lib" -L"$source_dir/buffertile/lib" \
  -L"$source_dir/tilestore/lib" "${blas_rpath[@]}" \
  -lexec -llam -lchunk -lbf -ltilestore -lopenblas -lm -lrt -lpthread \
  -o "$prefix/bin/adapter"
gcc -std=gnu11 -O2 -Wall -Wextra -fPIC -shared \
  "$script_dir/memfd_shm.c" -ldl -lpthread \
  -o "$prefix/bin/memfd_shm.so"
printf 'PreVision %s adapter: %s\n' "$commit" "$prefix/bin/adapter"
