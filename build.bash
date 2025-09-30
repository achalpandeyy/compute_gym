#!/bin/bash

target=$1

if [ -z "$target" ]; then
    echo "No build target specified"
    exit 1
fi

build_dir="build" && mkdir -p ${build_dir}
pushd $build_dir > /dev/null 2>&1

virtual_arch="compute_89"
real_arch="sm_89"

# TODO(achal): Add a flag to disable asserts.
# NOTE(achal): Passing -DNDEBUG to nvcc.exe disables host side asserts as well.

nvcc.exe \
-I "../../../cccl/" \
-I "../../../cccl/libcudacxx/include" \
-I "../../../cccl/cub" \
-I "../../../cccl/thrust" \
--use-local-env \
-std=c++17 \
-lineinfo \
--generate-code=arch=${virtual_arch},code="[${real_arch},${virtual_arch}]" \
-O3 \
-Xptxas=-v \
-Xcompiler "/std:c++17 /O2 /Oi /Ot" \
../"${target}".cu \
-o "${target}".exe

popd > /dev/null 2>&1