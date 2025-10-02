#ifndef CORE_MEMORY_H
#define CORE_MEMORY_H

#if defined(ARCH_X64)
#define PageSize (KiloBytes(4))
#elif defined(ARCH_ARM64) || defined(ARCH_WASM32)
#define PageSize (KiloBytes(64))
#endif

typedef struct Arena Arena;

static Arena *g_scratch_arena = 0;

#if PLATFORM_WINDOWS
inline static u8 *OS_MemoryReserve(u64 size)
{
	u8 *memory = (u8 *)VirtualAlloc(0, size, MEM_RESERVE, PAGE_READWRITE);
	return memory;
}

inline static void OS_MemoryCommit(void *memory, u64 size)
{
	VirtualAlloc(memory, size, MEM_COMMIT, PAGE_READWRITE);
}

// NOTE(achal): My experiments show that calling VirtualAlloc twice - first with MEM_RESERVE and next with MEM_COMMIT - is much slower than calling it once with both the flags, hence the existence of this function.
inline static u8 *OS_MemoryReserveAndCommit(u64 size)
{
    u8 *memory = (u8 *)VirtualAlloc(0, size, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE);
    return memory;
}

static void OS_MemoryDecommit(void *memory, u64 size)
{
	int retval = VirtualFree(memory, size, MEM_DECOMMIT);
	Assert(retval);
}

static void OS_MemoryRelease(void *memory, u64 size)
{
	VirtualFree(memory, size, MEM_RELEASE);
}
#elif PLATFORM_LINUX || PLATFORM_MACOS
#include <sys/mman.h>
inline static u8 *OS_MemoryReserve(u64 size)
{
	u8 *memory = (u8 *)mmap(0, size, PROT_NONE, MAP_PRIVATE|MAP_ANON, -1, 0);
	Assert((void *)memory != MAP_FAILED);
	return memory;
}

static void OS_MemoryCommit(void *memory, u64 size)
{
	mprotect(memory, size, PROT_READ|PROT_WRITE);
}

inline static u8 *OS_MemoryReserveAndCommit(u64 size)
{
    u8 *result = (u8 *)mmap(0, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);
	Assert((void *)result != MAP_FAILED);
    return result;
}

static void OS_MemoryDecommit(void *memory, u64 size)
{
	madvise(memory, size, MADV_DONTNEED);
	mprotect(memory, size, PROT_NONE);
}

static void OS_MemoryRelease(void *memory, u64 size)
{
	munmap(memory, size);
}
#endif

#if defined(ARCH_X64) || defined(ARCH_ARM64)
#include <string.h> // memset, memcpy

struct Arena
{
	u64 cap;
	u64 committed;
	u64 offset;
    
    // NOTE(achal): This points to the Arena struct itself which is followed
    // by the data it allocates i.e. the data really starts at
    // base + sizeof(Arena).
    //
	u8 *base;
};

// NOTE(achal):
// 1. `size` includes the size for the Arena struct itself.
// 2. `size` will get rounded up to next page boundary.
inline static Arena *InitArena(u64 size)
{
	u64 page_count = (size + PageSize - 1)/PageSize;
	size = page_count*PageSize;
    
	u8 *memory = OS_MemoryReserve(size);
	
	if (!memory) // @Investigate: How to handle errors here?
	{
		Assert(0);
		return 0;
	}
    
	u64 initial_commit_size = PageSize;
	Assert(initial_commit_size >= sizeof(Arena));
    
	OS_MemoryCommit(memory, initial_commit_size);
    
	Arena *arena = (Arena *)memory;
	arena->cap = size;
	arena->committed = initial_commit_size;
	arena->offset = sizeof(Arena);
	arena->base = memory;
    
	return arena;
}

inline static Arena *InitArenaDefault()
{
	u64 size = GigaBytes(8);
	Arena *result = InitArena(size);
	return result;
}

inline static u8 *GetArenaPos(Arena *arena)
{
    u8 *pos = arena->base + arena->offset;
    return pos;
}

inline static void ClearArena(Arena *arena)
{
	arena->offset = sizeof(Arena);
}

inline static void ReleaseArena(Arena *arena)
{
    // NOTE(achal): Since the address space for the entire arena was reserved by a single call to
    // VirtualAlloc, calling VirtualFree with zero will make sure that entire arena->cap worth of
    // address space gets released.
	OS_MemoryRelease(arena, 0);
}

static u8 *PushBytes(Arena *arena, u64 size)
{
	if (size == 0)
		return 0;
    
	u64 new_offset = arena->offset + size;
    
	if (new_offset > arena->cap)
	{
		Assert(!"Arena capacity reached!");
		return 0;
	}
    
	if (new_offset > arena->committed)
	{
		u64 size_to_commit = new_offset - arena->committed;
		u32 page_count = (u32)((size_to_commit + PageSize - 1) / PageSize);
		// TODO(achal): Handle failure.
		OS_MemoryCommit(arena->base + arena->committed, page_count*PageSize);
		arena->committed += page_count * PageSize;
	}
    
	u8 *result = arena->base + arena->offset;
	arena->offset = new_offset;
	return result;
}

static u8 *PushBytesZero(Arena *arena, u64 size)
{
	u8 *result = PushBytes(arena, size);
	memset(result, 0, size);
	return result;
}

static void PopBytesTo(Arena *arena, u64 offset)
{
	Assert(offset < arena->committed);
	u64 size_to_decommit = arena->committed - offset;
	u32 page_count = (u32)(size_to_decommit/PageSize);
	if (page_count)
	{
		arena->committed -= page_count * PageSize;
		OS_MemoryDecommit(arena->base + arena->committed, page_count*PageSize);
	}
    
	arena->offset = offset;
}

typedef struct
{
    Arena *arena;
    u64 offset;
} Scratch;

inline static Scratch ScratchBegin(Arena *arena)
{
    Scratch scratch = {arena, arena->offset};
    return scratch;
}

inline static void ScratchEnd(Scratch *scratch)
{
    scratch->arena->offset = scratch->offset;
}

static void MemoryCopy(u8 *dst, u8 *src, u64 size)
{
    memcpy(dst, src, size);
}
#elif defined(ARCH_WASM32)
// NOTE(achal):
// Making ArenaChunk fixed size means that when we get allocations that are larger than
// the size of the chunk we will have to allocate more than one ArenaChunk for a single
// allocation. Since, in general, ArenaChunks are not contiguous, we can't expect to
// allocation to be contiguous as well -- which is a guarantee Arena provides.
typedef struct ArenaChunk ArenaChunk;
struct ArenaChunk
{
    ArenaChunk *next;
    ArenaChunk *prev;
    
    u64 size;
    u64 offset;
    u8 *base;
};

static ArenaChunk *g_free_chunks_first = 0;
static ArenaChunk *g_free_chunks_last = 0;

struct Arena
{
    ArenaChunk *first_chunk;
    ArenaChunk *last_chunk;
    
    u64 size;
    u64 cap;
};

static u8 *PushBytes(Arena *arena, u64 size)
{
    if (size == 0)
		return 0;
    
    if (arena->size + size > arena->cap)
    {
        // TODO(achal): Assert(!"Arena capacity reached!");
        return 0;
    }
    
    ArenaChunk *chunk = 0;
    {
        chunk = arena->last_chunk;
        if (!chunk || (chunk->offset + size > chunk->size))
        {
            chunk = 0;
            // grab a new chunk
            for (ArenaChunk *c = g_free_chunks_first; c != g_free_chunks_last; c = c->next)
            {
                if (c->offset + size < c->size)
                {
                    chunk = c;
                    {
                        if (chunk == g_free_chunks_first)
                        {
                            g_free_chunks_first = chunk->next;
                        }
                        else if (chunk == g_free_chunks_last)
                        {
                            g_free_chunks_last = chunk->prev;
                        }
                        else
                        {
                            chunk->prev->next = chunk->next;
                        }
                    }
                    break;
                }
            }
            
            if (!chunk)
            {
                // NOTE(achal): This is to make sure that we don't allocate lots of small chunks. The choice
                // of PageSize is arbitrary
                u64 min_chunk_size = PageSize;
                u64 allocation_size = Maximum(size, min_chunk_size);
                
                u8 *memory = Platform_MemoryCommit(allocation_size);
                if (!memory)
                {
                    // TODO(achal): Handle failure?
                    Assert(0);
                }
                chunk = (ArenaChunk *)memory;
                chunk->prev = 0;
                chunk->next = 0;
                chunk->size = allocation_size;
                chunk->offset = sizeof(ArenaChunk);
                chunk->base = memory;
            }
            DoublyLinkedList_Push(arena->first_chunk, arena->last_chunk, next, prev, chunk);
        }
    }
    Assert(chunk);
    
    // allocate from the chunk
    u8 *result = chunk->base + chunk->offset;
    chunk->offset += size;
    arena->size += size;
    
    return result;
}

typedef struct
{
    Arena *arena;
    u64 size;
    ArenaChunk *chunk;
    u64 offset;
} Scratch;

inline static Scratch ScratchBegin(Arena *arena)
{
    Scratch scratch = {arena, arena->size, arena->last_chunk, arena->last_chunk->offset};
    return scratch;
}

inline static void ScratchEnd(Scratch *scratch)
{
    for (ArenaChunk *c = scratch->arena->last_chunk; c != scratch->chunk; c = c->prev)
    {
        c->offset = sizeof(ArenaChunk);
        scratch->arena->size -= c->size;
        DoublyLinkedList_Push(g_free_chunks_first, g_free_chunks_last, next, prev, c);
    }
    scratch->arena->last_chunk = scratch->chunk;
    scratch->arena->last_chunk->offset = scratch->offset;
    scratch->arena->size = scratch->size;
}

static void MemoryCopy(u8 *dst, u8 *src, u64 size)
{
    // @Optimization(Speed): Can we do better? For amd32 and arm32 we can just include
    // string.h, but 32-bit support right now is only for wasm and I don't know if there
    // is a good way for this without including all of standard library and making your
    // wasm module super bloated.
    for (u64 i = 0; i < size; ++i)
        *dst++ = *src++;
}
#endif

inline static Arena *GetScratchArena(u64 size)
{
    if (!g_scratch_arena)
        g_scratch_arena = InitArena(size);
    
    Assert(g_scratch_arena->cap >= size);
    return g_scratch_arena;
}

#define PushStruct(arena, type) (type *)PushBytes(arena, sizeof(type))
#define PushArray(arena, type, count) (type *)PushBytes(arena, count*sizeof(type))
#define PushArrayZero(arena, type, count) (type *)PushBytesZero(arena, count*sizeof(type))
#define PushStructZero(arena, type) (type *)PushBytesZero(arena, sizeof(type))

#endif // CORE_MEMORY_H
