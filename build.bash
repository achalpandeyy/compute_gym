#!/bin/bash

for arg in "$@"; do declare $arg="1"; done

if [ ! -v clang ];   then msvc=1;  fi
if [ ! -v release ]; then debug=1; fi

if [ -v msvc ];    then echo "[msvc compile]";  fi
if [ -v clang ];   then echo "[clang compile]"; fi
if [ -v debug ];   then echo "[debug mode]";    fi
if [ -v release ]; then echo "[release mode]";  fi
if [ -v reduce ];  then echo "reduce"; fi
if [ -v scan ];    then echo "scan";   fi

build_dir="build" && mkdir -p ${build_dir}
pushd $build_dir > /dev/null 2>&1

msvc_common="/std:c++17 /Zc:preprocessor"
msvc_debug="$msvc_common /Od /Zi /D BUILD_DEBUG"
msvc_release="$msvc_common /O2 /Oi /Ot /DNDEBUG"

if [ -v debug ]; then host_compile="$msvc_debug"; fi
if [ -v release ]; then host_compile="$msvc_release"; fi

virtual_arch="compute_89"
real_arch="sm_89"

# NOTE(achal): Defining macros for NVCC defines them for the host side as well.
# NOTE(achal): -keep to keep the intermediate files (including PTX) in the build directory.
# NOTE(achal): --ptxas-options=-v
nvcc_common="nvcc.exe -lineinfo -I ../../../cccl/ -I ../../../cccl/libcudacxx/include -I ../../../cccl/cub -I ../../../cccl/thrust --use-local-env -std=c++17 --generate-code=arch=${virtual_arch},code=[${real_arch},${virtual_arch}] --diag-suppress 186 -Xcompiler \"$host_compile\""
nvcc_debug="$nvcc_common -O0"
nvcc_release="$nvcc_common -O3"

if [ -v debug ];   then compile="$nvcc_debug";   fi
if [ -v release ]; then compile="$nvcc_release"; fi
out="-o"

ctime="../../../tools/ctime/ctime.exe"


if [ -v reduce ];     then $ctime -begin .reduce.ctm;     eval "$compile ../reduce.cu               $out reduce";     $ctime -end .reduce.ctm $?;     fi
if [ -v scan ];       then $ctime -begin .scan.ctm;       eval "$compile ../scan.cu                 $out scan";       $ctime -end .scan.ctm $?;       fi
if [ -v matmul ];     then $ctime -begin .matmul.ctm;     eval "$compile ../matmul.cu cublas.lib    $out matmul";     $ctime -end .matmul.ctm $?;     fi

# NOTE(achal): STREAM benchmark to measure peak sustained bandwidth.
if [ -v CUDAStream ]; then $ctime -begin .CUDAStream.ctm; eval "$compile ../ext/BabelStream/main.cu      $out CUDAStream"; $ctime -end .CUDAStream.ctm $?; fi

popd > /dev/null 2>&1