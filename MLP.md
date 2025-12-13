# Baseline

B = 4;
S = 64
M = B * S;
K = 768;
N = 3072;

## [0] `NVIDIA GeForce RTX 5060 Ti`
* SM Version: 1200 (PTX Version: 1200)
* Number of SMs: 36
* SM Default Clock Rate: 2602 MHz
* Global Memory: 15711 MiB Free / 15848 MiB Total
* Global Memory Bus Peak: 448 GB/sec (128-bit DDR @14001MHz)
* Max Shared Memory: 100 KiB/SM, 48 KiB/Block
* L2 Cache Size: 32768 KiB
* Maximum Active Blocks: 24/SM
* Maximum Active Threads: 1536/SM, 1024/Block
* Available Registers: 65536/SM, 65536/Block
* ECC Enabled: No

## OG

`TILE_SIZE`: 32

```
Run:  [1/1] bench_mlp_forward [Device=0 S=64]
Pass: Cold: 0.661276ms GPU, 0.675241ms CPU, 0.50s total GPU, 0.57s total wall, 757x 
Pass: Batch: 0.627479ms GPU, 0.52s total GPU, 0.52s total wall, 830x
```

| Samples |  CPU Time  | Noise |  GPU Time  | Noise | GlobalMem BW | BWUtil | Samples | Batch GPU  |
|---------|------------|-------|------------|-------|--------------|--------|---------|------------|
|    757x | 675.241 us | 0.27% | 661.276 us | 0.22% |  20.236 GB/s |  4.52% |    830x | 627.479 us |


## Warp tiled (2x2 micro tiling)

`MLP_TILE`: 16

```
Run:  [1/1] bench_mlp_forward [Device=0 S=64]
Pass: Cold: 0.409992ms GPU, 0.425268ms CPU, 0.50s total GPU, 0.61s total wall, 1220x 
Pass: Batch: 0.404001ms GPU, 0.52s total GPU, 0.52s total wall, 1285x
```

| Samples |  CPU Time  | Noise |  GPU Time  | Noise | GlobalMem BW | BWUtil | Samples | Batch GPU  |
|---------|------------|-------|------------|-------|--------------|--------|---------|------------|
|   1220x | 425.268 us | 0.36% | 409.992 us | 0.27% |  32.639 GB/s |  7.28% |   1285x | 404.001 us |

```
627.479/404.001 = 1.5531619971237698
```

## Warp tiled (2x2 micro tiling) + Double buffered

`MLP_TILE`: 32

```
Run:  [1/1] bench_mlp_forward [Device=0 S=64]
Pass: Cold: 0.309092ms GPU, 0.323859ms CPU, 0.50s total GPU, 0.65s total wall, 1618x 
Pass: Batch: 0.309478ms GPU, 0.52s total GPU, 0.52s total wall, 1672x
```

| Samples |  CPU Time  | Noise |  GPU Time  | Noise | GlobalMem BW | BWUtil | Samples | Batch GPU  |
|---------|------------|-------|------------|-------|--------------|--------|---------|------------|
|   1618x | 323.859 us | 0.45% | 309.092 us | 0.25% |  43.293 GB/s |  9.66% |   1672x | 309.478 us |

```
627.479/309.478=2.0275399220623114
```