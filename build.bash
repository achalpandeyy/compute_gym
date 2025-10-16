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

# NOTE(achal): Passing -DNDEBUG to nvcc.exe disables host side asserts as well.

../../../tools/ctime/ctime.exe -begin .${target}.ctm

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
--diag-suppress 186 \
-Xcompiler "/std:c++17 /O2 /Oi /Ot /Zi /Zc:preprocessor" \
../"${target}".cu \
-o "${target}".exe

../../../tools/ctime/ctime.exe -end .${target}.ctm $?

popd > /dev/null 2>&1