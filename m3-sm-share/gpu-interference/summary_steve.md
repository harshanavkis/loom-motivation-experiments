# device=NVIDIA H200 NVL sms=132 cc=9.0 gemm=8192x4096x7168 gemm_down=8192x7168x2048 bf16 triad_bytes=1073741824 occ_cluster=2

Copy-engine check (copy completes while all SMs are held => it runs on a copy engine):
- cudaMemcpyAsync d2d: completed_while_all_132_SMs_held=1
- cudaMemcpyAsync d2h: completed_while_all_132_SMs_held=1
- cudaMemcpyBatchAsync d2d: completed_while_all_132_SMs_held=1
- cudaMemcpyBatchAsync d2h: completed_while_all_132_SMs_held=1

### gemm: baseline 811.6 TFLOP/s (median of 5)

Throughput as % of the baseline; `ideal` = (SMs-k)/SMs; smcopy/ce columns give achieved comm GB/s in brackets.

| k | ideal | target | idle | smcopy d2d 50 | smcopy d2d 100 | smcopy d2d max | smcopy d2h 50 |
|---|---|---|---|---|---|---|---|
| 4 | 97.0 | 99.7 | 100.0 | 100.0 [50] | 99.9 [100] | 99.9 [117] | 34.2 [27] |
| 8 | 93.9 | 90.3 | 90.5 | 90.5 [50] | 90.4 [100] | 90.4 [232] | 20.5 [27] |
| 16 | 87.9 | 88.8 | 72.7 | 72.7 [50] | 72.6 [100] | 72.4 [454] | 4.6 [27] |
| 20 | 84.8 | 79.1 | 48.9 | 48.9 [50] | 48.9 [100] | 48.6 [574] | 2.1 [27] |
| 32 | 75.8 | 75.1 | 65.4 | 65.5 [50] | 65.4 [100] | 64.9 [809] | 2.2 [27] |

Copy engine moving the same bytes (no SMs held):

| dir | target GB/s | % of baseline | achieved GB/s |
|---|---|---|---|
| d2d | 50 | 99.9 | 50 |
| d2d | 100 | 99.9 | 83 |
| d2d | max | 99.9 | 83 |
| d2h | 50 | 99.8 | 29 |
| d2h | max | 99.8 | 29 |

### gemm_down: baseline 764.0 TFLOP/s (median of 5)

Throughput as % of the baseline; `ideal` = (SMs-k)/SMs; smcopy/ce columns give achieved comm GB/s in brackets.

| k | ideal | target | idle | smcopy d2d 50 | smcopy d2d 100 | smcopy d2d max | smcopy d2h 50 |
|---|---|---|---|---|---|---|---|
| 4 | 97.0 | 99.7 | 99.6 | 99.6 [50] | 99.3 [100] | 99.6 [114] | 41.5 [27] |
| 8 | 93.9 | 91.9 | 91.7 | 91.6 [50] | 91.6 [100] | 91.4 [225] | 25.7 [27] |
| 16 | 87.9 | 87.1 | 64.0 | 64.0 [50] | 63.9 [100] | 63.4 [464] | 3.7 [27] |
| 20 | 84.8 | 87.1 | 54.3 | 54.3 [50] | 54.1 [100] | 53.8 [568] | 2.8 [27] |
| 32 | 75.8 | 77.9 | 70.1 | 69.9 [50] | 69.7 [100] | 68.9 [796] | 2.4 [27] |

Copy engine moving the same bytes (no SMs held):

| dir | target GB/s | % of baseline | achieved GB/s |
|---|---|---|---|
| d2d | 50 | 99.6 | 51 |
| d2d | 100 | 99.8 | 81 |
| d2d | max | 99.6 | 81 |
| d2h | 50 | 99.6 | 29 |
| d2h | max | 99.7 | 29 |

### triad: baseline 3991.1 GB/s (median of 5)

Throughput as % of the baseline; `ideal` = (SMs-k)/SMs; smcopy/ce columns give achieved comm GB/s in brackets.

| k | ideal | idle | smcopy d2d 50 | smcopy d2d 100 | smcopy d2d max | smcopy d2h 50 |
|---|---|---|---|---|---|---|
| 4 | 97.0 | 97.2 | 76.0 [50] | 69.3 [62] | 69.2 [62] | 31.2 [27] |
| 8 | 93.9 | 96.8 | 95.8 [50] | 74.6 [100] | 68.3 [122] | 20.3 [27] |
| 16 | 87.9 | 93.5 | 94.4 [50] | 92.0 [100] | 66.2 [234] | 11.7 [27] |
| 20 | 84.8 | 92.6 | 95.9 [50] | 89.2 [100] | 65.0 [288] | 9.1 [27] |
| 32 | 75.8 | 92.0 | 93.9 [50] | 91.8 [100] | 62.4 [446] | 4.7 [27] |

Copy engine moving the same bytes (no SMs held):

| dir | target GB/s | % of baseline | achieved GB/s |
|---|---|---|---|
| d2d | 50 | 97.6 | 50 |
| d2d | 100 | 96.9 | 67 |
| d2d | max | 96.9 | 67 |
| d2h | 50 | 99.3 | 29 |
| d2h | max | 99.3 | 29 |

### Copy-engine throughput vs request size (scattered destinations)

| size (B) | copies | batch d2d GB/s | batch d2h GB/s | loop d2d GB/s | loop d2h GB/s | batch d2d Mcopies/s |
|---|---|---|---|---|---|---|
| 512 | 16384 | 0.6 | 0.8 | 0.2 | 0.2 | 1.18 |
| 1024 | 16384 | 1.6 | 1.7 | 0.4 | 0.5 | 1.52 |
| 2048 | 16384 | 3.1 | 3.3 | 0.9 | 1.0 | 1.53 |
| 3584 | 16384 | 5.3 | 5.6 | 1.5 | 1.7 | 1.48 |
| 7168 | 16384 | 10.6 | 10.0 | 3.0 | 2.9 | 1.48 |
| 14336 | 16384 | 18.4 | 14.8 | 5.8 | 5.5 | 1.29 |
| 65536 | 4096 | 46.5 | 22.8 | 24.6 | 14.9 | 0.71 |
| 262144 | 1024 | 70.9 | 27.2 | 89.9 | 23.9 | 0.27 |
| 1048576 | 256 | 81.6 | 28.6 | 304.9 | 27.5 | 0.08 |
| 16777216 | 16 | 85.6 | 29.0 | 1431.5 | 28.9 | 0.01 |
