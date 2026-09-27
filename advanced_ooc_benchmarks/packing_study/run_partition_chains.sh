#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
systemds_root=${SYSTEMDS_ROOT:-${script_dir}/../../../systemds}
java_bin=${JAVA_BIN:-/opt/devcon/env/java/current/bin/java}
cd "$systemds_root"
exec "$java_bin" -Xms512m -Xmx3g -XX:ActiveProcessorCount=4 --add-modules=jdk.incubator.vector \
  -cp 'target/test-classes:target/classes:target/lib/*' \
  org.apache.sysds.test.component.ooc.PartitionChainStudyTest \
  "${1:-lmcg}" "${2:-arrival}" "${3:-8000000}" "${4:-16}" "${5:-16777216}" "${6:-3}" "${7:-30}" "${8:-2}" "${9:-chunk}" "${10:-42}"
