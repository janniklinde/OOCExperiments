#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
systemds_root=${SYSTEMDS_ROOT:-${script_dir}/../../../systemds}
java_bin=${JAVA_BIN:-java}
cd "$systemds_root"

if [[ ! -f target/test-classes/org/apache/sysds/test/component/ooc/PackingMaterializerTest.class ]]; then
  echo 'Build first: mvn -q -Dtest=PackingMaterializerTest,PartitionedStoreTest,RowPackedMatrixOpsTest test' >&2
  exit 1
fi

exec "$java_bin" -Xms512m "-Xmx${PACKING_HEAP:-3g}" \
  "-XX:ActiveProcessorCount=${PACKING_CPUS:-4}" --add-modules=jdk.incubator.vector \
  -cp 'target/test-classes:target/classes:target/lib/*' \
  org.apache.sysds.test.component.ooc.PackingMaterializerTest \
  "${1:-8000000}" "${2:-67108864}" "${3:-32}" "${4:-32}" "${5:-4}" "${6:-4194304}"
