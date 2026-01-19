## CPU Optimization

For now this only covers GEMV optimization, currently working on GEMM writeup.

CPU: i5-7600K CPU @ 3.80GHz
I'm profiling with [perf](https://perfwiki.github.io/main/) 

```console
g++ -std=c++17 -O3 -g -fno-omit-frame-pointer -o llama_main src/*.cpp
perf stat -e cycles,instructions,cache-misses ./llama_main --tiny
```

first version:
Performance counter stats for './llama_main --tiny 20':

6,369,559,149    cycles                                                                
5,680,240,739    instructions                     #    0.89  insn per cycle            
165,210,482      cache-misses                                                          

1.519746884 seconds time elapsed

1.482653000 seconds user
0.037016000 seconds sys

We're looking at instructions per cycle (IPC) and cache-misses per 1000 instructions (MPKI) to understand what's happening. IPC is pretty self-explanatory, it's the number of instructions the CPU is executing in once clock cycle. My processor runs at ~3.8GHz = 3.8 billion cycles per second, so a runtime of 1.52 seconds makes sense.

On an i5‑7600K (Kaby Lake/Skylake‑class), a rough ceiling for IPC is 4 instructions per cycle. That usually means:

< 1.0 IPC: memory‑bound or branch‑heavy.
1–2 IPC: mixed workload, some stalls.
2–3.5 IPC: good compute‑bound kernels.
~4 IPC: very tight, well‑optimized, mostly in L1.

Our rate of 0.89 IPC is pretty bad! There are a few reasons this can happen, but a common culprit is cache misses. A CPU cache is a small, very fast memory that stores frequently used data or instructions so the CPU doesn't have to retrieve them in RAM. It's organized in levels (L1, L2, L3) where L1 is smallest and fastest (1 cache per CPU core), L2 is larger/slower (sometimes shared across cores), and L3 is largest/slowest (shared across all cores).

A cache miss happens when a CPU goes to look in a cache for a memory address, but doesn't find what it is looking for, so it needs to to fetch it from a slower level. (i.e., the CPU looks at L1 -> L2 -> L3 -> RAM)

Miss rate = cache-misses / cache-references. This tells you what fraction of cache accesses miss.
Broadly:

MPKI < 1: very low (cache‑friendly).
MPKI 1–5: moderate.
MPKI > 10: high, likely memory‑bound.

In general, IPC is what we're trying to optimize, and cache misses are one indicator of what could be improved. If you have low IPC and low cache misses, then your issue lies somewhere else (e.g., poor vectorization).

We have lots of cache misses, so we'll start by fixing that. We will start by creating a tiled (or blocked) matrix multiplication. The idea behind tiled matmul is that there’s a lot of data that we reuse can between computations, so we can design our matrix multiplication to use the data we've already fetched (in different ways), instead of blindly grabbing whatever data we need for the next computation. 

There are two main ideas in tiling: spatial locality (using *all* data the CPU has filled the cache with) and temporal locality (keeping the data we'll be using again soon cached). There are also two main kinds of matrix multiplication a LLM does: matrix by matrix, and vector by matrix. It does the former when it is processing an input (i.e., building the KV cache from a prompt), and the latter when it is generating an output (i.e., responding to your prompt).

It so happens that spatial locality is more useful for generating outputs, and temporal locality for processing inputs. Let's look at examples of matrix multiplication to illustrate why.

We'll start with vector-matrix multiplication. Let's look at a toy example, where A is a 1 x 26 row vector, A = [a, b, ..., z], and B is a matrix of dimension 26 x 2, B = [1, 27] // [2, 28] // ... // [26, 52]. We want to calculate A * B = C, a 1 x 2 row vector.

We'll calculate C iteratively, so we only need to load entries of A once: and we calculate C = [a * 1, a * 27], and so on, and finally load z to obtain our final result, C = [a * 1 + b * 2 + ... + z * 26, a * 27 + b * 28, ..., z * 52]. 

This doesn't seem too bad, until you look at how data is actually fetched in memory. The CPU doesn't fetch one element at a time, it fetches a cache *line*, which is (typically) a string of 64 bytes, or 16 floats. So if you ask for B[1][1], the CPU loads B[1][1], B[1][2], ..., and B[1][16] to the L1 cache.

Let's go back to our algorithm. At the first step, we ask the CPU for three objects in memory: a, 1, and 27. a is part of the vector A, which is small, so it's already fully stored in the L1 cache. Next we want "1", so the CPU loads 1, ..., 16, then we want 27, so the CPU loads 27, ..., 42. We'd be moving much faster if we were using data we loaded alongside 1, and this is the core idea behind GEMV: use all the data you're fetching in a cached line.

With this in mind, here is our GEMV implementation:

```cpp
// GEMV loop
// initialize output tensor
Tensor C({N});

// size of tile
const int BN = 64;

// per-tile loop, which is why we increment by tile size
for (int j0 = 0; j0 < N; j0 += BN) {

    // take min in case our vector length is not a multiple of the tile size
    // e.g., without min, for a vector of length 100, we'd try to increment up to 128, which is OOB
    int j_max = std::min(j0 + BN, N);

    // initialize output tile
    for (int j = j0; j < j_max; ++j) {
        C.data[j] = 0.0f; 
    }

    // temporal reuse: A's entries show up repeatedly, so 
    // we load each one and use it many times
    for (int k = 0; k < K; ++k) {
        const float a = A.data[k];

        // get the address of the B row tile we're using
        // we've gotten through k rows of size N, and we're on the j0th tile of that row
        const float* b_row_tile = &B.data[k * N + j0];

        // spatial locality: consume cache lines of B efficiently
        for (int j = j0; j < j_max; ++j) {
            C.data[j] += a * b_row_tile[j - j0]; // start at B row tile 0 and increment to max
        }
    }
}
return C;
```

Our hard work has been rewarded!

```console
Performance counter stats for './llama_main --tiny 20':

2,403,343,290      cycles                                                                
4,055,702,234      instructions                     #    1.69  insn per cycle            
72,395,863      cache-misses                                                          

0.571760942 seconds time elapsed

0.535494000 seconds user
0.036033000 seconds sys
```

IPC ≈ 1.69 (up from ~0.89 earlier)
MPKI ≈ 72,395,863 / 4,055,702,234 * 1000 ≈ 12.8 (down from ~29)

So: higher IPC and much lower MPKI tell us we've got better cache behavior, exactly what we wanted from the tiled GEMV.

For 2x2 multiplication / prefill
```console
perf stat -e cycles,instructions,cache-misses ./llama_main --tiny 20 --prefill 120
```

```cpp

const int BM = 64, BN = 64, BK = 64;

// matrices are stored row-major as
// A_{i,k} := A[i*K + k]
for (int i0 = 0; i0 < M; i0 += BM) {
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int j0 = 0; j0 < N; j0 += BN) {
            int i_max = std::min(i0 + BM, M);
            int k_max = std::min(k0 + BK, K);
            int j_max = std::min(j0 + BN, N);

            // i goes from i0 to i0 + BM (~0-64)
            for (int i = i0; i < i_max; ++i) {

                // k goes from k0 to k0 + BK (~0-64)
                for (int k = k0; k < k_max; ++k) {

                    // select an entry in the current A block
                    float a = A[i*K + k];

                    // addresses for 
                    const float* b_row = &B[k*N + j0];
                    float* c_row = &C[i*N + j0];

                    //
                    for (int j = j0; j < j_max; ++j) {
                        c_row[j - j0] += a * b_row[j - j0];
                    }
                }
            }
        }
    }
}

```
