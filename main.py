from profiler import WorkloadProfiler
from mapper import WorkloadMapper
from roofline import RooflineAnalyzer

if __name__ == '__main__':
    prof = WorkloadProfiler()
    mapper = WorkloadMapper()
    analyzer = RooflineAnalyzer()
    layers = prof.run_profiler()
    mapped = mapper.map_workload(layers)
    analyzer.generate_energy_plot(mapped)
    print('Plots generated.')
