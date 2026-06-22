from profiler import WorkloadProfiler
from mapper import WorkloadMapper

if __name__ == '__main__':
    prof = WorkloadProfiler()
    mapper = WorkloadMapper()
    layers = prof.run_profiler()
    mapped = mapper.map_workload(layers)
    print('Layer ID | NML Energy (J) | CMOS Energy (J)')
    for l in mapped:
        print(f"{l['layer_id']} | {l['nml_total_energy_J']} | {l['cmos_total_energy_J']}")
