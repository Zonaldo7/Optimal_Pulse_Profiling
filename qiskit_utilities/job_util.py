import functools
import paramiko
import signal
import time
import pickle
import os
import warnings
import pickle
import asyncio
import multiprocessing
from copy import deepcopy
from datetime import datetime
import threading
import numpy as np
import dill  

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJob, Session
from qiskit_ibm_runtime.runtime_job_v2 import RuntimeJobV2
from qiskit_ibm_runtime.ibm_backend import IBMBackend

from qiskit.providers import JobStatus
from qiskit.result import Result

import logging
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
# from .cr_pulse import get_default_cr_params

logger = logging.getLogger(__name__)
token = "REPLACE WITH IBM TOKEN"


class TimeoutError(Exception):
    pass


def _run_func(func, args, kwargs, result_queue):
    try:
        result = func(*args, **kwargs)
        result_queue.put(result)
    except Exception as e:
        result_queue.put(e)

def sanity_check(jobs):
    for job in jobs:
        job_finished = job.in_final_state()
        while not job_finished:
            time.sleep(15)
            job_finished = job.in_final_state()
        if job.status() != JobStatus.DONE and job.status() != "DONE":
            return False
    return True
        

def timeout_and_retry(timeout=30, reruns=3, verbose=True):
    """
    Decorator for adding timeout and retry functionality to a function.
    One should not use logging here because logging is not process-safe.

    Args:
        timeout (int, optional): Time limit for each attempt in seconds. Defaults to 30.
        reruns (int, optional): Number of retry attempts. Defaults to 3.
        verbose (bool, optional): Whether to print messages during retries. Defaults to True.
    """

    def decorator(func):
        func = deepcopy(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # return func(*args, **kwargs)
            for _ in range(reruns):
                manager = multiprocessing.Manager()
                result_queue = manager.Queue()

                # Run the function in a separate process
                process = multiprocessing.Process(
                    target=_run_func, args=(func, args, kwargs, result_queue)
                )
                process.start()

                # Wait for the function to finish
                process.join(timeout=timeout)

                if process.is_alive():
                    # The function did not finish within the timeout
                    process.terminate()
                    process.join()
                    if verbose:
                        print("Times out, restarting.")
                    continue

                result = result_queue.get()
                if isinstance(result, Exception):
                    raise result
                else:
                    return result

            raise TimeoutError(f"Action exceeded timeout after {reruns} attempts.")

        return wrapper

    return decorator


get_job_status = timeout_and_retry(timeout=10, reruns=8 * 60 * 60 * 2, verbose=False)(
    lambda job: job.done()
)


def get_single_job_result(job):
    """
    Get the result of Qiskit job, avoid unpickling lambda functions.
    """
    return job.result()

def get_job_result(job):
    """
    Get the result of a Qiskit job, waiting for its completion.

    Args:
        job (QiskitJob): Qiskit job object.

    Returns:
        Any: Result of the Qiskit job.
    """
    job_finished = job.done()
    while not job_finished:
        time.sleep(30)
        job_finished = job.done()
    result = timeout_and_retry(timeout=30, reruns=5, verbose=True)(
        get_single_job_result
    )(job)
    return result

def save_parallel_job_data(jobs, backend=None, parameters=None, amend=False, exp=None, rb=False):
    def process_single_result(result):
        if rb:
            return {"data": {"counts": result.data.meas.get_counts()},
                "shots": result.data.meas.num_shots,
                "success": True}
        return {"data": {"counts": result.data.c.get_counts()},
                "shots": result.data.c.num_shots,
                "success": True}
    def parallel_process_runtime_result(job):
        new_dict = {"results": []}
        results = job.result() 

        with ThreadPoolExecutor() as executor:
            processed_results = list(executor.map(process_single_result, results))
        
        new_dict["results"].extend(processed_results)
        new_dict["backend_name"] = job.backend().name
        new_dict["backend_version"] = job.backend().version
        new_dict["qobj_id"] = ''
        new_dict["job_id"] = job.job_id()
        new_dict["success"] = job.done()
        return new_dict
    # parallel_process_runtime_result(job)
    # job = service.job(job)
    for job in jobs:
        appendix = ".pickle"
        if not amend:
            if not os.path.exists("data"):
                os.makedirs("data")
            while os.path.exists("data/" + job.job_id() + appendix):
                appendix = "_tmp" + appendix
                warnings.warn(
                    f"File name {job.job_id()} exsits. Add a '_tmp' to the name.",
                    RuntimeWarning,
                )
        file_name = "data/" + job.job_id() + appendix
        parameters = dict() if parameters is None else parameters
        if not amend:
            data = {
                "parameters": parameters,
            }
            if exp is not None:
                # Due to a bug in qiskit runtime IBMBackend, expdata cannot be pickled.
                data["exp"] = exp
            with open(file_name, "wb") as f:
                # Save the parameter first to avoid losing it if the submission fails.
                pickle.dump(data, f)
        else:
            if parameters:
                warnings.warn("Parameters are not saved when amending a file.")
            with open(file_name, "rb") as f:
                data = pickle.load(f)
    
        job_finished = job.in_final_state()
        while not job_finished:
            time.sleep(15)
            job_finished = job.in_final_state()
        if job.status() != JobStatus.DONE and job.status() != "DONE":
            raise RuntimeError(
                f"Something went wrong with job {job.job_id()}: {job.status()}"
            )
        result = job.result()
        data.update(
            {
                "_result_dict": parallel_process_runtime_result(job),
                "job_id": job.job_id(),
            }
        )
        with open(file_name, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Job saved to data/{job.job_id()}\n")

    
    


def save_job_data(job, backend=None, parameters=None, amend=False, exp=None, rb=False):
    """
    Save the job data to a pickle file under the data folder.

    Args:
        job (Union[str, QiskitJob]): Qiskit job or job id.
        backend (Union[IBMBackend, QiskitRuntimeService], optional): IBMBackend or IBMQ Service.
            Needed if only the job id is provided. Defaults to None.
        parameters (dict, optional): A dictionary for parameters to be
            stored with the job. Defaults to None.
        amend (bool, optional): If True, open the existing file and
            store the result only. The parameters will not be saved.
            Defaults to False.
        exp (QiskitExperiment, optional): Qiskit experiment object.
            This only works for an experiment with a single job,
            and the experimental data will be generated automatically and
            saved in the parameter dictionary. Defaults to None.
    """
    def process_single_result(result):
        if rb:
            return {"data": {"counts": result.data.meas.get_counts()},
                "shots": result.data.meas.num_shots,
                "success": True}
        return {"data": {"counts": result.data.c.get_counts()},
                "shots": result.data.c.num_shots,
                "success": True}
    def parallel_process_runtime_result(job):
        new_dict = {"results": []}
        results = job.result() 

        with ThreadPoolExecutor() as executor:
            processed_results = list(executor.map(process_single_result, results))
        
        new_dict["results"].extend(processed_results)
        new_dict["backend_name"] = job.backend().name
        new_dict["backend_version"] = job.backend().version
        new_dict["qobj_id"] = ''
        new_dict["job_id"] = job.job_id()
        new_dict["success"] = job.done()
        return new_dict 

    if isinstance(job, str):
        if backend is None:
            raise ValueError(
                "If job ID is provided, the corresponding backend must also be provided."
            )
        if isinstance(backend, IBMBackend):
            job = backend.job(job)
        elif backend.version == 2:
            job = backend.provider.retrieve_job(job)
        elif backend.version == 1:
            job = backend.retrieve_job(job)
        else:
            raise ValueError("Unknown backend version.")

    appendix = ".pickle"
    if not amend:
        if not os.path.exists("data"):
            os.makedirs("data")
        while os.path.exists("data/" + job.job_id() + appendix):
            appendix = "_tmp" + appendix
            warnings.warn(
                f"File name {job.job_id()} exsits. Add a '_tmp' to the name.",
                RuntimeWarning,
            )
    file_name = "data/" + job.job_id() + appendix
    parameters = dict() if parameters is None else parameters
    if not amend:
        data = {
            "parameters": parameters,
        }
        if exp is not None:
            # Due to a bug in qiskit runtime IBMBackend, expdata cannot be pickled.
            data["exp"] = exp
        with open(file_name, "wb") as f:
            # Save the parameter first to avoid losing it if the submission fails.
            pickle.dump(data, f)
    else:
        if parameters:
            warnings.warn("Parameters are not saved when amending a file.")
        with open(file_name, "rb") as f:
            data = pickle.load(f)

    job_finished = job.in_final_state()
    while not job_finished:
        time.sleep(15)
        job_finished = job.in_final_state()
    if job.status() != JobStatus.DONE and job.status() != "DONE":
        raise RuntimeError(
            f"Something went wrong with job {job.job_id()}: {job.status()}"
        )
    if isinstance(job, RuntimeJobV2):
        result = job.result()
    else:
        result = get_job_result(job)
    if isinstance(job, RuntimeJobV2):
        data.update(
            {
                "_result_dict": parallel_process_runtime_result(job),
                "job_id": job.job_id(),
            }
        )
    else:
        data.update(
            {
                "_result_dict": result.to_dict(),
                "job_id": job.job_id(),
            }
        )

    with open(file_name, "wb") as f:
        pickle.dump(data, f)
    # if isinstance(job, RuntimeJob):
    #     tags = job.tags
    # else:
    #     tags = job.tags()
    logger.info(f"Job saved to data/{job.job_id()}\n")


def load_job_data(job: str, blocking=False, path="data"):
    """
    Load job data from a pickle file.

    Args:
        job (Union[str, QiskitJob]): Qiskit job or job id.
        blocking (bool, optional): If True, wait for the job to complete before returning. Defaults to False.
        path (str, optional): Path to the data folder. Defaults to "data".

    Returns:
        dict: Loaded job data.
    """
    if not isinstance(job, str):
        job_id = job.job_id()
    else:
        job_id = job

    def _acquire_data(job_id):

        with open(path + "/" + job_id + ".pickle", "rb") as f:
            data = pickle.load(f)
        #####################
        # It turns out that pickling circuits is not stable, so we remove them when we can.
        if "circuits" in data:
            del data["circuits"]
        if "inputs" in data:
            if "circuits" in data["inputs"]:
                del data["inputs"]["circuits"]
        if "result" in data:
            data["_result_dict"] = data["result"].to_dict()
            del data["result"]
        with open(path + "/" + job_id + ".pickle", "wb") as f:
            pickle.dump(data, f)
        #####################
        if "_result_dict" in data:
            data["result"] = Result.from_dict(data["_result_dict"])
        return data

    data = _acquire_data(job_id)
    if "result" not in data:
        if not blocking:
            logging.warning(
                "No result fund, job may not have finished yet. Use blocking=true to wait for it to complete."
            )
        else:
            time.sleep(15)
            while "result" not in data:
                data = _acquire_data(job_id)
    return data


def _log_error(func):
    """
    A decorator to log the error raised in the separate thread.
    Typically, in asynchronous execution, the error will not be raised until ``await`` is called.
    Thus we write any exception to the log file and raise it again.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(e)
            raise e

    return wrapper


def async_execute(func, *args, **kwargs):
    """
    Create a separate thread to execute the function without blocking the current thread.Only returns an async handle object, need to use ``await`` to retrieve the results, which will block until the result is available.

    Args:
        func (callable): A Python function.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        asyncio.Task: An asyncio coroutine object representing the asynchronous execution of the function.
    """
    func = _log_error(func)
    handle = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    return handle


def execute_job(runner, *args, **kwargs):
    """
    Execute a job using the provided runner, either a Qiskit Runtime session or a Qiskit QuantumProvider.

    Args:
        runner (QiskitRuntimeSession or IBMQProvider): Qiskit Runtime session or QuantumProvider.
        **kwargs: Keyword arguments to be passed to the job execution.

    Returns:
        Job: The executed job.
    """
    if isinstance(runner, Session):
        job = runner.run(
            "circuit-runner", inputs=kwargs, options={"max_execution_time": 300}
        )
    else:
        runner.run(*args, **kwargs)
    return job


def retrieve_expdata(job, backend, exp):
    """
    Retrieve qiskit-experiment data associated with a job ID and backend.

    Args:
        job (str): Job ID of the executed job.
        backend (IBMQBackend): The backend used for the job.

    Returns:
        ExperimentData: Experimental data associated with the job.
    """
    expdata = exp._initialize_experiment_data()
    if isinstance(backend, QiskitRuntimeService):
        expdata.add_jobs([backend.job(job)])
    elif isinstance(job, str) and backend.name == "DynamicsBackend":
        raise ValueError("For DynamicsBackend, please provides the job instance.")
    elif backend.name == "DynamicsBackend":
        expdata.add_jobs([job])
    else:
        expdata.add_jobs([backend.job(job)])
    expdata = exp.analysis.run(expdata).block_for_results()
    return expdata


def _get_amp_omega_GHz_ratio(backend, qubit_ind):
    if backend.name == "DynamicsBackend":
        return 2 * np.pi
    x_pulse = (
        backend.target.instruction_schedule_map()
        .get("x", qubit_ind)
        .instructions[0][1]
        .pulse
    )
    x_gate_duration_ns = x_pulse.duration * backend.dt
    default_drive_vac = x_pulse.amp
    # We assume the relation 2pi*ratio*amp*t/2 = 1/2 for x gate
    # Where ratio * amp should be the omega we use in the theory.
    omega_Hz = 1 / x_gate_duration_ns
    omega_GHz = omega_Hz * 1e-9
    return default_drive_vac / omega_GHz


def amp_to_omega_GHz(backend, qubit_ind, amp):
    """
    Convert pulse amplitude to qubit frequency in GHz.

    Args:
        backend (IBMQBackend): The backend on which the pulse is applied.
        qubit_ind (int): Index of the qubit.
        amp (float): Amplitude of the pulse.

    Returns:
        float: Qubit frequency in GHz.
    """
    ratio = _get_amp_omega_GHz_ratio(backend, qubit_ind)
    return amp / ratio


def omega_GHz_to_amp(backend, qubit_ind, omega_GHz):
    """
    Convert qubit frequency in GHz to pulse amplitude.

    Args:
        backend (IBMQBackend): The backend on which the pulse is applied.
        qubit_ind (int): Index of the qubit.
        omega_GHz (float): Qubit frequency in GHz.

    Returns:
        float: Amplitude of the pulse.
    """
    ratio = _get_amp_omega_GHz_ratio(backend, qubit_ind)
    return omega_GHz * ratio


def save_calibration_data(backend, gate_name, qubits, qubit_calibration_data):

    backend_name = backend if isinstance(backend, str) else backend.name
    with threading.Lock():
        # Preventing the file being read by different threads at the same time.

        if os.path.exists("calibration_data.pickle"):
            with open("calibration_data.pickle", "rb") as f:
                calibration_data = pickle.load(f)
        else:
            calibration_data = {}
        qubit_calibration_data["date"] = datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        calibration_data[(backend_name, gate_name, qubits)] = qubit_calibration_data
        with open("calibration_data.pickle", "wb") as f:
            pickle.dump(calibration_data, f)


def read_calibration_data(backend, gate_name, qubit_ind):

    with open("calibration_data.pickle", "rb") as f:
        calibration_data = pickle.load(f)
    backend_name = backend if isinstance(backend, str) else backend.name
    return calibration_data[(backend_name, gate_name, qubit_ind)]

def process_single_rb_result(job, shots, lengths, samples, num_qubits=2):
    def process_single_result(result):
        return {"data": {"counts": result.data.meas.get_counts()},
                "shots": result.data.meas.num_shots,
                "success": True}
    def parallel_process_runtime_result(job):
        new_dict = {"results": []}
        results = job.result() 

        with ThreadPoolExecutor() as executor:
            processed_results = list(executor.map(process_single_result, results))
        
        new_dict["results"].extend(processed_results)
        new_dict["backend_name"] = job.backend().name
        new_dict["backend_version"] = job.backend().version
        new_dict["qobj_id"] = ''
        new_dict["job_id"] = job.job_id()
        new_dict["success"] = job.done()
        return new_dict 
    def moder_0th_order(m, a0, p, b0):
        return a0 * p**m + b0   
    def moder_1th_order(m, a0, p, b0, c0):
        return a0 * p**m + c0 * (m - 1) * p**(m-2) + b0
    result_dict = parallel_process_runtime_result(job)
    rb_counts = []
    irb_counts = []
    for _ in range(len(result_dict.get("results"))):
        if _ % 2 == 0:
            rb_counts.append(result_dict.get("results")[_].get("data").get("counts").get("00") / shots)
        else:
            irb_counts.append(result_dict.get("results")[_].get("data").get("counts").get("00") / shots)
    lengths_plot = lengths*samples
    lengths_e = np.array(lengths)
    counts_e = [np.mean([rb_counts[_] for _ in range(i, len(lengths_plot), len(lengths)) ]) for i in range(len(lengths))]
    inter_e = [np.mean([irb_counts[_] for _ in range(i, len(lengths_plot), len(lengths)) ]) for i in range(len(lengths))]
    bounds_0 = ([-np.inf, 5e-4, -np.inf], [np.inf, 0.15, np.inf])
    bounds_1 = ([-np.inf, 5e-4, -np.inf, -np.inf], [np.inf, 0.15, np.inf, np.inf])
    popt1_0, pcov1_0 = curve_fit(moder_0th_order, lengths_e, counts_e, maxfev=10000)
    popt2_0, pcov2_0 = curve_fit(moder_0th_order, lengths_e, inter_e, maxfev=10000)
    try:
        popt1_1, pcov1_1 = curve_fit(moder_1th_order, lengths_e, counts_e, maxfev=10000)
        res1_1 = np.sum((moder_1th_order(lengths_e, *popt1_1) - counts_e)**2)
    except:
        res1_1 = np.inf
    try:
        popt2_1, pcov2_1 = curve_fit(moder_1th_order, lengths_e, inter_e, maxfev=10000)
        res2_1 = np.sum((moder_1th_order(lengths_e, *popt2_1) - inter_e)**2)
    except:
        res2_1 = np.inf
    res1_0 = np.sum((moder_0th_order(lengths_e, *popt1_0) - counts_e)**2)
    res2_0 = np.sum((moder_0th_order(lengths_e, *popt2_0) - inter_e)**2)
    # res1_1 = np.sum((moder_1th_order(lengths_e, *popt1_1) - counts_e)**2)
    # res2_1 = np.sum((moder_1th_order(lengths_e, *popt2_1) - inter_e)**2)
    if res1_0 < res1_1:
        popt = popt1_0
        pcov = pcov1_0
        plot_func1 = moder_0th_order
    else:
        popt = popt1_1
        pcov = pcov1_1
        plot_func1 = moder_1th_order

    
    if res2_0 < res2_1:
        popt2 = popt2_0
        pcov2 = pcov2_0
        plot_func2 = moder_0th_order
    else:
        popt2 = popt2_1
        pcov2 = pcov2_1
        plot_func2 = moder_1th_order

    d = 2 ** num_qubits
    r = (d-1)*(1-popt2[1]/popt[1])/d
    error1 = ((d-1)*(np.abs(popt[1]-popt2[1]/popt[1]) + 1 - popt[1])) / d
    error2 = 2 * (d**2 - 1) * (1 - popt[1]) / ( popt[1] * d**2) \
        + 4 * np.sqrt(1 - popt[1]) * np.sqrt(d**2 - 1) / (popt[1])
    error_range = np.min([error1, error2])
    print("Fidelity = [", r - error_range, ", ", r + error_range, "]")
    print("Fitting error for RB = ", np.sqrt(np.diag(pcov)))
    print("Fitting error for IRB = ", np.sqrt(np.diag(pcov2)))
    fig, ax = plt.subplots()
    ax.scatter(lengths_plot, rb_counts, label="RB")
    ax.scatter(lengths_plot, irb_counts, label="IRB")
    lengths_plot = np.arange(0, lengths_e[-1], 0.1)
    ax.plot(lengths_plot, plot_func1(lengths_plot, *popt), label="RB fit")
    ax.plot(lengths_plot, plot_func2(lengths_plot, *popt2), label="IRB fit")
    ax.legend()
    return r, error_range

def get_total_time(job_list):
    total_time = 0
    service = QiskitRuntimeService(token=token, channel="ibm_quantum") 
    for job in job_list:
        job = service.job(job)
        total_time += job.usage_estimation["quantum_seconds"]
    return total_time


def send_tele_msg(
        msg,
):
    hostname = "REPLACE WITH HOSTNAME"
    port = 22
    username = "username"
    passkey = "passkey"
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname, port, username, key_filename=passkey)
    stdin, stdout, stderr = ssh_client.exec_command(f"python3 ~/py_scripts/send_start.py --msg \"{msg}\"")
    ssh_client.close()