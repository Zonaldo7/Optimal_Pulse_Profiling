import numpy as np
import rustworkx as rx
import matplotlib.pyplot as plt
import threading
from copy import deepcopy
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit

from qiskit_experiments.framework import (
    ParallelExperiment, 
    BatchExperiment, 
    ExperimentData,
)
from qiskit.circuit.library import ECRGate
from qiskit_experiments.library import StandardRB, InterleavedRB
from qiskit_device_benchmarking.bench_code.prb import PurityRB
from qiskit.transpiler import InstructionProperties
from qiskit.compiler import transpile
from qiskit_device_benchmarking.utilities import graph_utils as gu
from .cr_cnot import (
    create_echoed_cr_schedule,
    create_direct_cnot_schedule,
    create_direct_cr_schedule
)
from .job_util import (
    read_calibration_data,
    send_tele_msg
)
import logging
logger = logging.getLogger(__name__)
import paramiko
import pickle
import os
import polars as pl
from datetime import datetime
from qiskit.circuit import Gate
from qiskit.circuit.library import ECRGate, CXGate
from qiskit.pulse import ScheduleBlock
from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler



    
def gen_sep_sets(
        backend,
        min_sep=2
):
    nq = backend.configuration().n_qubits
    coupling_map = backend.configuration().coupling_map
    G = gu.build_sys_graph(nq, coupling_map)
    paths = rx.all_pairs_all_simple_paths(G, 2, 2)
    paths = gu.paths_flatten(paths)
    paths = gu.remove_permutations(paths)
    paths = gu.path_to_edges(paths, coupling_map)
    sep_sets = gu.get_separated_sets(G, paths, min_sep)
    return sep_sets

def decompose_custom_ecr(
        sched:ScheduleBlock,
):
    half_sched_1 = ScheduleBlock()
    half_sched_2 = ScheduleBlock()
    half_duration = sched.instructions[0][1].duration
    for instruction in sched.instructions:
        if instruction[0] <= half_duration:
            half_sched_1.append(instruction[1])
        else:
            half_sched_2.append(instruction[1])
    return half_sched_1, half_sched_2
    



def gen_parallel_rb(
        qubit_pairs,
        lengths,
        num_sampling,
        shots,
        backend,
        gate_name,
):
    # return
    backend_tmp = deepcopy(backend)
    target = backend_tmp.target
    possible_qubit_pairs = deepcopy(qubit_pairs)
    if gate_name == "CR-default":
        duration_lst = []
        for qubit_pair in qubit_pairs:
            calibration_data = read_calibration_data(
                backend_tmp,
                gate_name,
                tuple(qubit_pair),
            )
            custom_ecr = create_echoed_cr_schedule(
                backend_tmp,
                tuple(qubit_pair),
                calibration_data,
                reverse_direction=False,
                parallel_exp=True,
            )
            duration_lst.append(custom_ecr.duration)
            target["ecr"][tuple(qubit_pair)] = \
                InstructionProperties(calibration=custom_ecr)
    elif gate_name == "CR-machine":
        pass
    elif gate_name == "CR-recursive-tr10":
        ecr_half_1 = Gate(name="ecr_half_1", num_qubits=2, params=[])
        ecr_half_2 = Gate(name="ecr_half_2", num_qubits=2, params=[])
        ecr_half_1_prop = {}
        ecr_half_2_prop = {}
        decompostion = QuantumCircuit(2)
        decompostion.append(ecr_half_1, [0, 1])
        decompostion.append(ecr_half_2, [0, 1])
        ecr_gate_custom = ECRGate().to_mutable()
        ecr_gate_custom.add_decomposition(decompostion)
        for qubit_pair in qubit_pairs:
            calibration_data = read_calibration_data(
                backend_tmp,
                gate_name,
                tuple(qubit_pair),
            )
            try:
                custom_ecr_sched = create_echoed_cr_schedule(
                    backend=backend_tmp,
                    qubits=tuple(qubit_pair),
                    calibration_data=calibration_data,
                    reverse_direction=False,
                    parallel_exp=True,
                )
                # print(qubit_pair, "successful")
                ecr_half_1_sched, ecr_half_2_sched = \
                    decompose_custom_ecr(custom_ecr_sched)
                ecr_half_1_prop[tuple(qubit_pair)] = \
                    InstructionProperties(calibration=ecr_half_1_sched)
                ecr_half_2_prop[tuple(qubit_pair)] = \
                    InstructionProperties(calibration=ecr_half_2_sched)
            except:
                # print(qubit_pair)
                possible_qubit_pairs = deepcopy(possible_qubit_pairs)
                possible_qubit_pairs.remove(qubit_pair)
                # print(possible_qubit_pairs)
                # print(qubit_pairs)
                continue
            
        # print(possible_qubit_pairs)
        # print("ECR half 1 properties:", ecr_half_1_prop)
        # print("ECR half 2 properties:", ecr_half_2_prop)
        assert len(possible_qubit_pairs) == len(ecr_half_1_prop)
        # exit()
        target.add_instruction(ecr_half_1, ecr_half_1_prop)
        target.add_instruction(ecr_half_2, ecr_half_2_prop)
    
    elif gate_name == "CR-recursive-tr10-direct":
        custom_cnot = CXGate().to_mutable()
        cnot_prop = {}
        for qubit_pair in qubit_pairs:
            calibration_data = read_calibration_data(
                backend_tmp,
                gate_name,
                tuple(qubit_pair),
            )
            try:
                direct_cnot = create_direct_cnot_schedule(
                    backend_tmp,
                    tuple(qubit_pair),
                    calibration_data,
                    reverse_direction=False,
                    parallel_exp=True,
                )
                cnot_prop[tuple(qubit_pair)] = \
                    InstructionProperties(calibration=direct_cnot)
            except:
                possible_qubit_pairs = deepcopy(possible_qubit_pairs)
                possible_qubit_pairs.remove(qubit_pair)
                continue
        
        assert len(possible_qubit_pairs) == len(cnot_prop)
        target.add_instruction(custom_cnot, cnot_prop)
            # custom_ecr = 


    else:
        raise ValueError("Gate name not recognized.")
                
    rb_lst = []
    if gate_name == "CR-default" or gate_name == "CR-machine":
        rb_gate = ECRGate()
    elif gate_name == "CR-recursive-tr10":
        rb_gate = ecr_gate_custom
    elif gate_name == "CR-recursive-tr10-direct":
        rb_gate = custom_cnot
    else:
        raise ValueError("Gate name not recognized.")


    
    for qubit_pair in possible_qubit_pairs:
        seed = np.random.randint(2**16)
        exp = InterleavedRB(
            interleaved_element=rb_gate,
            physical_qubits=qubit_pair,
            lengths=lengths,
            backend=backend_tmp,
            num_samples=num_sampling,
            seed=seed,
        )
        rb_lst.append(exp)
    parallel_irb = ParallelExperiment(rb_lst,
                                      backend=backend_tmp,
                                      flatten_results=False)
    if gate_name == "CR-default" or gate_name == "CR-machine":
        parallel_irb_exp = BatchExperiment(
            [parallel_irb],
            backend=backend_tmp,
            flatten_results=False
        )
        parallel_irb_exp.set_experiment_options(max_circuits=100)
        parallel_irb_exp.set_run_options(shots=shots)
        parallel_irb_data = parallel_irb_exp.run()
        parallel_irb_exp_job_ids = parallel_irb_data.job_ids
        logger.info("Parallel IRB experiment job ids " + \
                    f"for qubits {possible_qubit_pairs}: " + \
                    f"{parallel_irb_exp_job_ids}")
        send_tele_msg("Parallel IRB experiment job ids " + \
                    f"for qubits {possible_qubit_pairs}: " + \
                    f"{parallel_irb_exp_job_ids}")
    
    elif gate_name == "CR-recursive-tr10":
        parallel_irb_exp = BatchExperiment(
            [parallel_irb],
            backend=backend_tmp,
            flatten_results=False
        )
        # return parallel_irb_exp.circuits(), backend_tmp, parallel_irb_exp
        transpiled_circ = transpile(
            parallel_irb_exp.circuits(),
            backend=backend_tmp,
            basis_gates=["x", "sx", "rz", "measure"] + \
                ["ecr_half_1", "ecr_half_2"],
            optimization_level=1,
            initial_layout=[qubit for qubit_pair in possible_qubit_pairs 
                            for qubit in qubit_pair],
        )
        jobs = []
        max_circuits = 10
        all_partitioned_circuits = []
        for i in range(0, len(transpiled_circ), max_circuits):
            all_partitioned_circuits.append(transpiled_circ[i : i + max_circuits])
        with Batch(backend=backend_tmp):
            sampler = Sampler()
            for partitioned_circuits in all_partitioned_circuits:
                jobs.append(sampler.run(partitioned_circuits, shots=shots))
        parallel_irb_exp.set_experiment_options(max_circuits=10)
        parallel_irb_exp.set_run_options(shots=shots)
        parallel_irb_exp_data = ExperimentData(experiment=parallel_irb_exp)
        parallel_irb_exp_data.add_jobs(jobs)
        parallel_irb_data = parallel_irb_exp.analysis.run(
            parallel_irb_exp_data
        ).block_for_results()
    
    elif gate_name == "CR-recursive-tr10-direct":
        parallel_irb_exp = BatchExperiment(
            [parallel_irb],
            backend=backend_tmp,
            flatten_results=False
        )
        # return parallel_irb_exp.circuits(), backend_tmp, parallel_irb_exp
        parallel_irb_exp.set_experiment_options(max_circuits=10)
        parallel_irb_exp.set_run_options(shots=shots)
        parallel_irb_data = parallel_irb_exp.run()
        logger.info("Parallel IRB experiment job ids " + \
                    f"for qubits {possible_qubit_pairs}: " + \
                    f"{parallel_irb_data.job_ids}")
        send_tele_msg("Parallel IRB experiment job ids " + \
                    f"for qubits {possible_qubit_pairs}: " + \
                    f"{parallel_irb_data.job_ids}")
        parallel_irb_data.block_for_results()

    else:
        raise ValueError("Gate name not recognized.")
    
    return parallel_irb_data, possible_qubit_pairs

def save_rb_data(
        rb_data,
        qubit_pairs,
        gate_name,
        lengths,
):
    if os.path.exists(f"rb_data_{gate_name}.pkl"):
        with open(f"rb_data_{gate_name}.pkl", "rb") as f:
            rb_data_dict = pickle.load(f)
    else:
        rb_data_dict = {}
    for i, qubit_pair in enumerate(qubit_pairs):
        rb_data_dict[tuple(qubit_pair)] = {}
        rb_data_dict[tuple(qubit_pairs[i])]["alpha"] = rb_data.child_data()[0].child_data()[i].analysis_results()[1].value.nominal_value
        rb_data_dict[tuple(qubit_pairs[i])]["alphas"] = rb_data.child_data()[0].child_data()[i].analysis_results()[1].value.std_dev
        rb_data_dict[tuple(qubit_pairs[i])]["alpha_c"] = rb_data.child_data()[0].child_data()[i].analysis_results()[2].value.nominal_value
        rb_data_dict[tuple(qubit_pairs[i])]["alpha_cs"] = rb_data.child_data()[0].child_data()[i].analysis_results()[2].value.std_dev
        rb_data_dict[tuple(qubit_pairs[i])]["epc"] = rb_data.child_data()[0].child_data()[i].analysis_results()[3].value.nominal_value
        rb_data_dict[tuple(qubit_pairs[i])]["epcs"] = rb_data.child_data()[0].child_data()[i].analysis_results()[3].value.std_dev
        rb_data_dict[tuple(qubit_pairs[i])]["job_ids"] = rb_data.job_ids
        rb_data_dict[tuple(qubit_pairs[i])]["exp_qubits"] = qubit_pairs
        rb_data_dict[tuple(qubit_pairs[i])]["lengths"] = lengths
        rb_data_dict[tuple(qubit_pairs[i])]["data"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"rb_data_{gate_name}.pkl", "wb") as f:
        pickle.dump(rb_data_dict, f)

def parallel_rb_exp(
    qubit_pairs,
    lengths,
    num_sampling,
    shots,
    backend,
    gate_name
):
    rb_data, qubit_pairs = gen_parallel_rb(
        qubit_pairs,
        lengths,
        num_sampling,
        shots,
        backend,
        gate_name,
    )
    while True:
        if 'successful' in rb_data.analysis_status().value:
            break
    send_tele_msg(f"Parallel IRB experiment for qubits {qubit_pairs} is done.")
    save_rb_data(
        rb_data,
        qubit_pairs,
        gate_name,
        lengths,
    )



def clear_calibration_time(
        gate_name,
        qubit_pairs=None
):
    with open("calibration_data.pickle", "rb") as f:
        calibration_data = pickle.load(f)
    for key in calibration_data.keys():
        if qubit_pairs is not None:
            if key[1] == gate_name and list(key[2]) in qubit_pairs:
                print(f"Clearing calibration time for {key}")
                calibration_data[key]["calibration_time"] = 0
        if key[1] == gate_name:
            calibration_data[key]["calibration_time"] = 0
    with threading.Lock():
        # Preventing the file being read by different threads at the same time.
        with open("calibration_data.pickle", "wb") as f:
            pickle.dump(calibration_data, f)
    