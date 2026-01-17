This is my implementation of Llama-3 in C++. It's part 2 of what I intend to be a 3 part series, covering a naive C++ implementation, optimization for CPU (blocked GEMM/GEMV, vectorization, multithreading...), and optimization for GPU (GEMM/vectorized matmul kernel, FlashAttention kernel...). You can find part 1 [here](https://github.com/etiennedyer/llama-cpp/tree/part-1).

You can get the real weights [here](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (about 15gb, BF16 safetensors, model-00001 to model-00004 + model.safetensors.index.json, behind an auth wall), and download them to ./weights/, or just run a small version with the --tiny flag.

## Repo structure

├── README.md
├── src
│   ├── layers.cpp      // defines RoPE, attention, transformer forward pass, etc.
│   ├── layers.h
│   ├── main.cpp        // execution loop, define parameters here
│   ├── model.cpp       // defines the full forward loop
│   ├── model.h
│   ├── safetensors.cpp // loads weights BF16 weights
│   ├── safetensors.h
│   ├── tensor.cpp      // tensor class, matmul (GEMM/GEMV)
│   └── tensor.h
└── weights             // empty, fill with Llama-3 8B weights

## Running

To run:
Compile with

```console
g++ -std=c++17 -O2 -o llama_main src/*.cpp 
```

run with
```console
./llama_main 
```

runtime flags: 
```
--tiny
```

runs the loop with placeholder matrices, with parameters specified in main.cpp.

```
--tiny N 
```
to run the loop for N iterations (defaults to 1 if N ommited)

```
--prefill T
```
run in prefill mode with random prefill of size T. Only available in tiny mode

*Thoughts on implementation: Part 2, Profiling*

Profiling:
CPU: i5-7600K CPU @ 3.80GHz
I'm profiling with perf (only available on linux)

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

We're looking at instructions per cycle (IPC) and cache-misses per 1000 instructions (MPKI) to understand what's happening. IPC is fairly self-explanatory, it's the number of instructions the CPU is executing in once clock cycle. My processor runs at ~3.8GHz = 3.8 billion cycles per second, so a runtime of 1.52 seconds makes sense.

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

## 2×3 matrix times 3×2 matrix

\( A [2, 3] \) times 
\( B [3, 2] \), producing \( C [2, 2] \).

where

\[
A =
\begin{bmatrix}
a & b & c \\
d & e & f
\end{bmatrix}
\]

\[
B =
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
\]

### Output
C is 2x2, with

\[
C_{1,1} = a1 + b2 + c3
\]

\[
C_{1,2} = a4 + b5 + c6
\]

\[
C_{2,1} = d1 + e2 + f3
\]

\[
C_{2,2} = d4 + e5 + f6
\]

Note how every input is used twice. This points to the main idea behind temporal use / GEMM: you'll be reusing inputs a lot, so keep them handy. 

*Example of GEMM*

Now let's look at vector-matrix multiplication.

## Example 2: 1×3 vector times 3×2 matrix

We multiply a row vector \( A \in \mathbb{R}^{1 \times 3} \) by a matrix  
\( B \in \mathbb{R}^{3 \times 2} \), producing \( C \in \mathbb{R}^{1 \times 2} \).

### Inputs

\[
A =
\begin{bmatrix}
a & b & ... & z
\end{bmatrix}
\]

\[
B =
\begin{bmatrix}
1 & 27 \\
2 & 28 \\
... \\
26 & 52
\end{bmatrix}
\]

### Output

We get a 1x2 matrix C, where

\[
C_{1,1} = a * 1 + b * 2 + z * 26
\]

\[
C_{1,2} = a * 27 + b * 28 + z * 52
\]

Here, every entry of A is used in both entries, so we still want to keep it handy. Thankfully, A is a vector, so it is inexpensive to store, even as it grows large. However, note that the entries of B only show up once. That means we won't save time by reusing data from B we've already loaded. 

Let's try our iterative algorithm from the first example: our first pass loads a, and we calculate C = [a * 1, a * 27], and so on, and finally load z to obtain our final result, C = [a * 1 + b * 2 + ... + z * 26, a * 27 + b * 28, ..., z * 52]. 

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