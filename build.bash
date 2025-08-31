#!/bin/bash

target=$1

if [ -z "$target" ]; then
    echo "No build target specified"
    exit 1
fi

build_dir="build" && mkdir -p ${build_dir}
pushd $build_dir > /dev/null 2>&1

nvcc.exe \
--use-local-env \
-lineinfo \
--generate-code=arch=compute_75,code="[sm_75,compute_75]" \
-O3 \
-DNDEBUG \
-Xptxas=-v \
-Xcompiler "/O2 /Oi /Ot /DNDEBUG" \
../"${target}".cu \
-o "${target}".exe

popd > /dev/null 2>&1