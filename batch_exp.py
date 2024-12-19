import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import rustworkx as rx
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit

from qiskit_experiments.framework import ParallelExperiment, BatchExperiment, ExperimentData

from qiskit_experiments.library import StandardRB
from qiskit_device_benchmarking.bench_code.prb import PurityRB

from qiskit_device_benchmarking.utilities import graph_utils as gu
import logging
from qiskit_utilities import (
    setup_logger,
    get_default_cr_params,
    parallel_cr_pulse_calib,
    gen_initial_data_lst,
    gen_sep_sets,
    parallel_rb_exp,
    parallel_rz0_correction_calibration,
    parallel_fine_rz0_correction_calibration,
    clear_calibration_time
)
setup_logger(filename=None, level=logging.INFO, stdout=True)
logger = logging.getLogger("qiskit_utilities")

token = "REPLACE WITH IBM TOKEN"
service = QiskitRuntimeService(token=token, channel="ibm_quantum")
backend=service.backend('ibm_rensselaer')
sep_sets = gen_sep_sets(backend, min_sep=2)
# print(sep_sets)
# for sep_set in sep_sets:
    # print(sep_set)

# batch_size = 40
max_size = 17
qubit_pairs_lst = []
lengths = [1, 10, 20, 50, 100, 150, 250, 400]
for ii in range(len(sep_sets)):
    qubit_pairs = sep_sets[ii]
    print(f"Qubit pairs set {ii}:")
    print(len(qubit_pairs))
    gate_name = "CR-recursive-tr10"

qubit_pairs = sep_sets[0]
# clear_calibration_time(gate_name, qubit_pairs=qubit_pairs)
# initial_calibration_data_lst, cr_times = \
#     gen_initial_data_lst(backend,
#                          qubit_pairs,
#                          gate_name)
# finished = parallel_cr_pulse_calib(
#     qubits=qubit_pairs,
#     backend=backend,
#     initial_calibration_data=initial_calibration_data_lst,
#     cr_times=cr_times,
#     gate_name=gate_name,
#     verbose=True,
#     restart=True,
#     threshold_MHz=0.015,
#     max_repeat=4,
#     shots=512,
#     mode="CR",
# )
# parallel_rb_exp(
#     finished,
#     lengths=lengths,
#     num_sampling=10,
#     shots=4096,
#     backend=backend,
#     gate_name=gate_name
# )













#####################
# gate_name = "CR-recursive-tr10-direct"
# clear_calibration_time(gate_name, qubit_pairs=qubit_pairs)
# initial_calibration_data_lst, cr_times = \
#     gen_initial_data_lst(backend, 
#                          qubit_pairs,
#                          gate_name)
# for i, qubit_pair in enumerate(qubit_pairs):
#     if initial_calibration_data_lst[i]["cr_params"]["amp"] > 1:
#         qubit_pairs.remove(qubit_pair)
#         logger.info(f"Qubit pair {qubit_pair} is removed due to the amplitude of CR pulse is larger than 1.")
# finished = parallel_cr_pulse_calib(
#     qubits=qubit_pairs,
#     backend=backend,
#     initial_calibration_data=initial_calibration_data_lst,
#     cr_times=cr_times,
#     gate_name=gate_name,
#     verbose=True,
#     restart=True,
#     threshold_MHz=0.015,
#     max_repeat=4,
#     shots=2048,
#     mode="CR",
#     rerun_last_calibration=True
# )
# QUBIT_C = qubit_pairs[0][0]
# QUBIT_T = qubit_pairs[0][1]
# # ratio = 2
# duration = backend.defaults().instruction_schedule_map.get("ecr", (QUBIT_C, QUBIT_T)).duration/16 * 4 / 2 / 2
# cr_times = 16 * np.arange(16, duration + 16, duration//30, dtype=int)
# finished = parallel_cr_pulse_calib(
#     qubits=finished,
#     backend=backend,
#     cr_times=cr_times,
#     initial_calibration_data=None,
#     verbose=True,
#     threshold_MHz=0.015,
#     restart=True,
#     max_repeat=3,
#     shots=8096,
#     mode="IX-pi",
#     gate_name=gate_name,
#     rerun_last_calibration=False
# )
# parallel_rz0_correction_calibration(
#     backend=backend,
#     qubit_pairs=finished,
#     gate_name=gate_name,
# )
# parallel_fine_rz0_correction_calibration(
#     backend=backend,
#     qubit_pairs=finished,
#     gate_name=gate_name,
#     max_repeat_cnot=6
# )
    
# parallel_rb_exp(
#     finished,
#     lengths=lengths,
#     backend=backend,
#     num_sampling=10,
#     shots=4096,
#     gate_name=gate_name
# )

