#ifndef CORE_H
#define CORE_H

inline static u32 FindMSB(u32 x)
{
#if COMPILER_MSVC
    unsigned long index;
	if (_BitScanReverse(&index, (unsigned long)x))
		return index;
	else
		return ~0u;
#elif COMPILER_CLANG
    if (x == 0)
		return 0;
	u32 result = (sizeof(x)*8 - 1) - __builtin_clz(x);
	return result;
#elif COMPILER_NVCC
    // TODO(achal): When using the nvcc compiler I probably use the host compiler here.
    Assert(!"TODO");
#endif
}

#define QueuePush(first, last, n)         \
{                                       \
Assert((n));                          \
\
void **p_first = (void **)(&(first)); \
void **p_last  = (void **)(&(last));  \
\
if ((first) == 0)                     \
{                                     \
Assert((last) == 0);                \
*p_first = *p_last = (n);           \
}                                     \
else                                  \
{                                     \
(last)->next = (n);                 \
*p_last = (n);                      \
}                                     \
(n)->next = 0;                        \
}                                       \

#define ListPush(first, last, next, n)    \
{                                       \
Assert((n));                          \
\
void **p_first = (void **)(&(first)); \
void **p_last  = (void **)(&(last));  \
\
if ((first) == 0)                     \
{                                     \
Assert((last) == 0);                \
*p_first = *p_last = (n);           \
}                                     \
else                                  \
{                                     \
(last)->next = (n);                 \
*p_last = (n);                      \
}                                     \
(n)->next = 0;                        \
}                                       \

#define DoublyLinkedList_Push(first, last, next, prev, n)  \
{                                                        \
Assert((n));                                           \
\
void **p_first = (void **)(&(first));                  \
void **p_last  = (void **)(&(last));                   \
if ((first) == 0)                                      \
{                                                      \
Assert((last) == 0);                                 \
*p_first = *p_last = (n);                            \
}                                                      \
else                                                   \
{                                                      \
Assert((last) != 0);                                 \
(last)->next = (n);                                  \
(n)->prev = last;                                    \
*p_last = (n);                                       \
}                                                      \
(n)->next = 0;                                         \
}


#define StackPush(top, n)  \
{                        \
(n)->next = (top);     \
(top) = (n);           \
}

#define StackPop(top)      \
{                        \
if (top)               \
{                      \
(top) = (top)->next; \
}                      \
}

#endif // CORE_H
