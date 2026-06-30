from profiler import WorkloadProfiler
from mapper import WorkloadMapper
from roofline import RooflineAnalyzer

if __name__ == '__main__':
    prof = WorkloadProfiler()
    mapper = WorkloadMapper()
    analyzer = RooflineAnalyzer()
    layers = prof.run_profiler()
    mapped = mapper.map_workload(layers)
    table = analyzer.run_sensitivity(mapped, 102.4e9, 204.8e9)
    print(table)
