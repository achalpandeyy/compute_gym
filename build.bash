#!/bin/bash

target=$1

if [ -z "$target" ]; then
    echo "No build target specified"
    exit 1
fi

build_dir="build" && mkdir -p ${build_dir}
pushd $build_dir > /dev/null 2>&1

nvcc.exe \
-I "../../../cccl/" \
-I "../../../cccl/libcudacxx/include" \
-I "../../../cccl/cub" \
-I "../../../cccl/thrust" \
--use-local-env \
-std=c++17 \
-lineinfo \
--generate-code=arch=compute_75,code="[sm_75,compute_75]" \
-O3 \
-Xptxas=-v \
-Xcompiler "/std:c++17 /O2 /Oi /Ot" \
../"${target}".cu \
-o "${target}".exe

popd > /dev/null 2>&1