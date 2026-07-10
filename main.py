import sys
from profiler import WorkloadProfiler
from mapper import WorkloadMapper

if __name__ == '__main__':
    prof = WorkloadProfiler(use_live_profiling=False)
    layers = prof.run_profiler()
    mapper = WorkloadMapper()
    mapped = mapper.map_workload(layers)
    print("SRAM Cache Hit Rate logged.")
