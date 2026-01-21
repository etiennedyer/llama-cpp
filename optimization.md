# Matmul Optimization

This is my writeup of the process of optimizing the CPU matrix multiplication for this codebase. So far, I've covered tiled GEMV and GEMM, and I'm currently doing SIMD vectorization. The last step after that will be multithreading, and then I'll move on to GPU kernels in CUDA.

I've put a lot of effort into making this as instructive as possible, and would gladly take any feedback or constructive criticism.

## Profiling

Before we get started with improving things, we need to establish a performance baseline.

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

```console
Performance counter stats for './llama_main --tiny 20 --prefill 120':

22,539,729,531      cycles                                                                
21,331,208,904      instructions                     #    0.95  insn per cycle            
160,628,103      cache-misses                                                          

5.354763764 seconds time elapsed

5.312370000 seconds user
0.042002000 seconds sys
```

Back to below 1 IPC, which is expected, since we're now using our naive algorithm to do 2x2 matmul.

As we did above, we'll create a toy example to illustrate the core ideas. (That being said, GEMM is a little more complicated than GEMV. It certainly took me longer to wrap my head around it.)

Say we want to do a 4x4 matmul. To simplify things a little, we’ll assume our fast memory can store 16 floats, and cache lines contain 4 floats. We’ll start by storing A and C in row-major format, then B in column-major format, so our cache lines efficiently retrieve data. (This changes though, so pay attention!)

The worst-case scenario, with no data reuse, requires 16 operations (1 per entry of C), with 3 retrievals each (1 row of A + 1 column of B + 1 row or column from C) = 48 retrievals. Note that this never really happens, because even a naive algorithm will end up accidentally reusing data — the data isn’t evicted from fast memory until something else needs to come up.

Let’s look at how data often naturally ends up being reused: we can compute our sums along a single row of C, in order to reuse the rows of A and C we’ve already loaded. This brings us down to 1 (row of A) + 4 (columns of B) + 1 (row of C) = 6 retrievals per row of C, of which we have 4, so 24 retrievals total.

We've been reusing the first row of A and C, which is helpful, but we'd like to reuse columns of B as well.  To do this, we’ll need another row of A — if we load the second row of A, that allows us to calculate C[2, 1] as well. Now we’ll store C as column-major to update 2x1 tiles.

That’s 2 (rows of A) + 1 ( column of B) + 1 (column of C) = 4 retrievals, and our 16-float cache is full.

We’ll calculate C by iterating over columns, to reuse our 2 rows of A. To calculate 2 rows of C, we load 1 x 2 rows of A, 4 x 1 columns of B, and 4x1 column of C = 10 retrievals. We do this twice to calculate all four rows, so we have 20 retrievals.

Now, we’re calculating 2 elements at once, but if we added another column of B, we could calculate 4. Unfortunately, our cache is already full. But notice we aren’t using it as well as we could be: we use full rows of A, but only half of the B column and half of the C column. Let’s pretend we can add a second column of B, then see how we can use our cache more efficiently so it fits.

The first thing we notice that we don't need to calculate an entire element of C at once. Take for instance:
c11 = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41. 

We could easily do this in two passes: 
c11 = a11 * b11 + a12 * b21, 
c11 += a13 * b31 + a14 * b41

We also wanted to calculate c12, which becomes:
c12 = a11 * b12 + a12 * b22, 
c12 += a13 * b32 + a14 * b42

Similar for c21 and c22.

Notice what data we’re using: in the first pass, we only need a11, a12, a21, a22 from A , b11, b21, b12, b22 from B, and c11, c12, c21, c22 from C. (In each case, this is the top-left 2x2 sub-matrix / quadrant.) This is only 12 floats, but they’re across different rows/columns, so how do we retrieve them efficiently? We can do something called matrix packing: we'll store our matrix so that a cache line retrieves a 2x2 sub-matrix. (Note that we're mainly packing for the purposes of this toy example -- this specific motivaiton of packing doesn't apply in our code, since our blocks are bigger than our cache lines. We'll go deeper into packing and motivating its use when we cover SIMD.)

Now, let’s think about how we reuse our cached data. If you go to the matmul visualizer tool and hover over C, you’ll notice the first pass over C only uses the left half of A, and the top half of B. We’ll use our 4 extra floats in memory to keep one of these “hot” while we change out the other. We’ll keep B hot and swap out A tiles, but we could have do 

1st pass:
Load A_TL, B_TL B_TR, C_TL. (+ 4 retrievals)
1. C_TL: use A_TL with B_TL.

2. C_TR: A_TL is also used with B_TR for C_TR, so we swap C_TL for C_TR. (+1)

3. C_BR: B_TR is also used for C_BR, so swap A_TL for A_BL, and C_TR for C_BR (+2)

4. B_TL is also used with A_BL for C_BL so swap C_BR for C_BL (+1)

2nd pass:
Load A_TR, B_BL B_BR, C_TL. (+ 4 retrievals)
1. C_TL: A_TR, B_BL       

2. C_TR: A_TR is also used with B_BR for C_TR, so we swap C_TL for C_TR. (+1)     

3. C_BR: B_BR is also used for C_BR, so swap A_TR for A_BR, and C_TR for C_BR. (+2)     

4. C_BL: B_BL is also used with A_BR for C_BL so swap C_BR for C_BL. (+1)

Total: 16 retrievals.

Note: we did not count the cost of writebacks here.  If you look at our first pass, when C_TL is read at step 1, it has value 0. In step 2, when it is replaced by C_TR, it has value A_TL x B_TL, so the CPU needs to go change its stored value in slow memory. (This only happens for C tiles, because we change their values — if we swap out a tile of A, its value hasn’t changed, so we don’t need to perform a write operation). If we added the cost of writebacks (8 in total), we’d be back to 24 read/write operations (not interchangeable, but close enough for this toy model ). I chose to ommit them here for illustrative purposes, because in real GEMM kernels you try hard to avoid repeatedly evicting partially-accumulated C like this by keeping tiles of C in cache until both passes are done—then you only need to pay one load + one store per C tile, not two rounds of load/evict. We could have done that here, but we would have had to reload B more often.

```cpp

// index for blocks of A/C
// controls rows of A and C
// i0 changes -> C and A row-tiles change.
for (int i0 = 0; i0 < M; i0 += BM) {
    int i_max = std::min(i0 + BM, M);

    // index for blocks of B/C
    // controls columns of B and C
    // j0 changes -> C and B column-tiles change.
    for (int j0 = 0; j0 < N; j0 += BN) {
        int j_max = std::min(j0 + BN, N);

        // index for blocks of A/B
        // controls rows of B, columns of A
        // k0 changes -> A and B tiles change, C tile stays the same.
        for (int k0 = 0; k0 < K; k0 += BK) {
            int k_max = std::min(k0 + BK, K);
            // this order keeps C hot:
            // because i0 and j0 are the outermost loops,
            // with (i0, j0) fixed, we iterate k0 and keep accumulating into the same C tile.
            // This tends to reduce C reloads/writebacks compared to loop orders that revisit a C tile once per k0.

            // for this kind of memory layout, you don't want i0 inside
            // because then C can get fully evicted, meaning more writebacks

            for (int i = i0; i < i_max; ++i) {
                // iterate over rows of C
                // in chunks of 64
                float* c_row = C.data.data() + i * N + j0; // matrices are stored row-major, A_{i,k} := A[i*K + k]
                // as i increases, move to a new row of C
                // as j0 increases, move to a new block of columns
                
                // iterate over the k (columns of of A / corresponding rows of B) in chunks of 64
                for (int k = k0; k < k_max; ++k) {

                    const float a = A.data[i * K + k];
                    // as k increases, move along the row of A (i.e., increase column)
                    // as i increases, move to a new row

                    // broadcast a into a vector in memory
                    __m256 a8 = _mm256_set1_ps(a);
                    
                    const float* b_row = B.data.data() + k * N + j0;
                    // as j0 increases, move to the next column-block of B 
                    // (i.e., row stays the same, but) you're touching different columns
                    // as k increases, move to a new row (stride N)
                    // k moves faster, so we fix a column and increase the row
                    // go for BK = 64 rows, then new block

                    // use a scalar in the stored row of A 
                    // while we move through columns of a row of B and C
                    for (int j = j0; j < j_max; ++j) {
                        c_row[j - j0] += a * b_row[j - j0];
                    }
                }
            }
        }
    }
}

```

## Vectorization

The fundamental idea behind the current AI boom is vectorization. When you're doing matrix multiplication, the entries of your output can be calculated fully independently, so you're free to do them in any order you like... or even at the same time. The code we've written so far does not use that at all: if you look at the innermost loop of our GEMM algorithm, we do one multiplication per cycle. My CPU, the Intel i5-7600K has two 128-bit SIMD lanes, meaning I can perform 8 operations involving 32-bit floats in parallel. 

```cpp
// vectorized multiplication helper
static inline void fma_row_update_8(__m256 a8, const float* b, float* c) {

    // load 8 floats from B
    // b8 = [b[0], b[1], ..., b[7]]
    __m256 b8 = _mm256_loadu_ps(b);

    // load 8 floats from C
    __m256 c8 = _mm256_loadu_ps(c);

    // fused multiply-add
    // c8[i] = a8[i] * b8[i] + c8[i]
    c8 = _mm256_fmadd_ps(a8, b8, c8);

    // store updated values back to C
    _mm256_storeu_ps(c, c8);
}
```

```cpp
for (int k = k0; k < k_max; ++k) {

    const float a = A.data[i * K + k];

    __m256 a8 = _mm256_set1_ps(a);
    const float* b_row = B.data.data() + k * N + j0;

    int j = j0;
    // use a scalar in the stored row of A 
    // while we move through columns of a row of B and C
    // vectorize to go 8 spots at a time
    for (; j + 8 <= j_max; j+=8) {
        fma_row_update_8(a8, b_row + (j - j0), c_row + (j - j0));
    }

    // finish up what isn't a multiple of 8
    for (; j < j_max; ++j) {
        c_row[j - j0] += a * b_row[j - j0];
    }
}
```

```console
Performance counter stats for './llama_main --tiny 20 --prefill 120':

3,439,443,967      cycles                                                                
7,007,935,038      instructions                     #    2.04  insn per cycle            
12,825,163      cache-misses                                                          

0.821737072 seconds time elapsed

0.781599000 seconds user
0.040082000 seconds sys
```