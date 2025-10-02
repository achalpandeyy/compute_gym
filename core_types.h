#ifndef CORE_TYPES_H
#define CORE_TYPES_H

#if defined(__clang__)
#define COMPILER_CLANG 1
#elif defined(_MSC_VER)
#define COMPILER_MSVC 1
#elif defined(__NVCC__)
#define COMPILER_NVCC
#else
#error "Compiler not supported"
#endif

#if defined(_M_X64) || defined(__x86_64__)
#define ARCH_X64 1
#elif defined(__aarch64__)
#define ARCH_ARM64 1
#elif defined(__wasm32__)
#define ARCH_WASM32 1
#else
#error "Architecture not supported"
#endif

#if defined(_WIN32)
#define PLATFORM_WINDOWS 1
#elif defined(__linux__)
#define PLATFORM_LINUX 1
#elif defined(__APPLE__)
#define PLATFORM_MACOS 1
#elif defined(__wasm32__)
#define PLATFORM_WASM 1
#else
#error "Platform not supported"
#endif

#define KiloBytes(x) (u64)(1024ull*(x))
#define MegaBytes(x) (u64)(1024ull*KiloBytes(x))
#define GigaBytes(x) (u64)(1024ull*MegaBytes(x))

#define Minimum(x, y) ((x) < (y) ? (x) : (y))
#define Maximum(x, y) ((x) > (y) ? (x) : (y))
#define ArrayCount(x) (sizeof(x)/sizeof((x)[0]))

#define CONCAT_IMPL(a, b) a##b
#define CONCAT(a, b) CONCAT_IMPL(a, b)

#if PLATFORM_WINDOWS
#define COBJMACROS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdint.h>

// TODO(achal): Platform-specific asserts.
#if !ARCH_WASM32
#include <assert.h>
#define Assert assert
#else
#define Assert(...)
#endif

#define StaticAssert(cond, label) u8 static_assert_##label[(cond) ? (1) : (-1)];

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef u8 b8;
typedef u32 b32;

typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef float f32;
typedef double f64;

#endif // CORE_TYPES_H
