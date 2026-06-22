from profiler import WorkloadProfiler
from mapper import WorkloadMapper

if __name__ == '__main__':
    prof = WorkloadProfiler()
    mapper = WorkloadMapper()
    layers = prof.run_profiler()
    mapped = mapper.map_workload(layers)
    print(f'Mapped {len(mapped)} layers.')
