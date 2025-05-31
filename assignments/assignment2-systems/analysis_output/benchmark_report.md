# Distributed Communication Benchmark Report

Generated on: 2025-05-31 16:11:44.907254

## Summary Statistics

### Performance by Configuration

                                  mean_time_ms        bandwidth_mb_per_sec         min_time_ms
                                          mean    std                 mean     std        mean
backend device world_size size_mb                                                             
gloo    cpu    2          1.0            0.665  0.040             1505.589  90.369       0.458
                          10.0           4.479  0.038             2232.958  18.963       3.956
               4          1.0            1.760  0.026              852.353  12.781       1.095
                          10.0           6.217  0.024             2412.908   9.293       5.028
                          100.0         58.040  0.058             2584.409   2.570      50.748

## Key Findings

### Backend Performance
- **GLOO**: 1929.7 MB/s average bandwidth

### Scaling Behavior
Average latency by process count:

**GLOO + CPU:**
- 2 processes: 2.57 ms
- 4 processes: 22.01 ms

## Analysis Notes

- All-reduce operations show expected scaling behavior
- Bandwidth generally increases with data size due to better amortization of latency
- NCCL+GPU typically outperforms Gloo+CPU for large data sizes
- Communication overhead becomes more significant with more processes
