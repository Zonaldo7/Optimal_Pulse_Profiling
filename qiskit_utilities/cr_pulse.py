"""
Generation and calibration of CR pulse.
"""

import os
import warnings
import multiprocessing as mpl
from multiprocessing import Pool
from functools import partial
from copy import deepcopy

import numpy as np
from numpy import pi

try:
    import jax
    import jax.numpy as jnp
except:
    warnings.warn("JAX not install, multi-derivative pulse doesn't work.")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
from concurrent.futures import ThreadPoolExecutor
import scipy
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import symengine as sym
import matplotlib.pyplot as plt

from qiskit import pulse, circuit, schedule, transpile
from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumCircuit
from qiskit.pulse.library import (
    Gaussian,
    GaussianSquare,
    GaussianSquareDrag,
    SymbolicPulse,
    Waveform,
)
from qiskit.pulse import Play, ShiftPhase, DriveChannel, ScheduleBlock, Schedule
from ._custom_jax import _value_and_jacfwd

from .job_util import (
    amp_to_omega_GHz,
    omega_GHz_to_amp,
    load_job_data,
    save_job_data,
    async_execute,
    read_calibration_data,
    save_calibration_data,
    save_parallel_job_data,
    get_total_time,
    sanity_check,
    send_tele_msg
)
from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler
# Logger setup after importing logging
import logging
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
# from pydantic import BaseModel

logger = logging.getLogger(__name__)

def gen_initial_data_lst(backend, qubit_pairs, gate_name, ratio=1):
    initial_calibration_data_lst = []
    if gate_name == "CR-default":
        for qubit_pair in qubit_pairs:
            cr_config, ix_config = get_default_cr_params(backend, 
                                                         qubit_pair[0],
                                                         qubit_pair[1])
            duration = backend.defaults().instruction_schedule_map.get(
                "ecr", (qubit_pair[0], qubit_pair[1])
            ).duration / 16 * 4
            cr_params = {
                "duration": int(duration),
                "sigma": 64,
                "amp": cr_config["amp"],
                "angle": cr_config["angle"],
            }
            ix_params = {
                "duration": int(duration),
                "sigma": 64,
                "amp": ix_config["amp"],
                "angle": ix_config["angle"],
            }
            initial_data = {
                "cr_params": cr_params,
                "ix_params": ix_params,
                "frequency_offset": 0,
            }
            initial_calibration_data_lst.append(initial_data)
    elif gate_name == "CR-recursive-tr10":
        for qubit_pair in qubit_pairs:
            QUBIT_C, QUBIT_T = qubit_pair[0], qubit_pair[1]
            cr_config, ix_config = get_default_cr_params(backend, 
                                                         qubit_pair[0],
                                                         qubit_pair[1])
            f0 = backend.properties().frequency(QUBIT_C)
            f1 = backend.properties().frequency(QUBIT_T)
            a0 = backend.properties().qubit_property(QUBIT_C)["anharmonicity"][0]
            a1 = backend.properties().qubit_property(QUBIT_T)["anharmonicity"][0]
            duration = backend.defaults().instruction_schedule_map.get(
                "ecr", (QUBIT_C, QUBIT_T)).duration / 16 * 4
            params = {
                "order": "3",
                "t_r": 10,
                "drag_type": "exact",
                "duration": duration,
                "amp": cr_config["amp"],
                "angle": cr_config["angle"],
                "Delta": (backend.properties().frequency(QUBIT_C) - backend.properties().frequency(QUBIT_T)) * 1.0e-9 * 2 * pi,
                "a1": 2 * pi * backend.properties().qubit_property(QUBIT_C)["anharmonicity"][0] * 1.0e-9,
                "drag_scale": [1., 1., 1.]
            }
            cr_params = deepcopy(params)
            ix_params = deepcopy(params)
            ix_params["amp"] = ix_config["amp"]
            ix_params["angle"] = ix_config["angle"]
            ix_params["order"] = "2"
            initial_data = {
                "cr_params": cr_params,
                "ix_params": ix_params,
                "frequency_offset": 0,
            }
            initial_calibration_data_lst.append(initial_data)
    elif gate_name == "CR-recursive-tr10-direct":
        for qubit_pair in qubit_pairs:
            ratio = 2
            cr_config, ix_config = get_default_cr_params(
                backend, qubit_pair[0], qubit_pair[1]
            )
            QUBIT_C, QUBIT_T = qubit_pair[0], qubit_pair[1]
            f0 = backend.properties().frequency(QUBIT_C)
            f1 = backend.properties().frequency(QUBIT_T)
            a0 = backend.properties().qubit_property(QUBIT_C)["anharmonicity"][0]
            a1 = backend.properties().qubit_property(QUBIT_T)["anharmonicity"][0]
            duration = backend.defaults().instruction_schedule_map.get(
                "ecr", (QUBIT_C, QUBIT_T)).duration / 16 * 4 / ratio
            params = {
                "order": "3",
                "t_r": 10,
                "drag_type": "exact",
                "duration": duration,
                "amp": cr_config["amp"] * ratio,
                "angle": cr_config["angle"],
                "Delta": (backend.properties().frequency(QUBIT_C) - backend.properties().frequency(QUBIT_T)) * 1.0e-9 * 2 * pi,
                "a1": 2 * pi * backend.properties().qubit_property(QUBIT_C)["anharmonicity"][0] * 1.0e-9,
                "drag_scale": [1., 1., 1.]
            }
            cr_params = deepcopy(params)
            ix_params = deepcopy(params)
            ix_params["amp"] = ix_config["amp"] * ratio
            ix_params["angle"] = ix_config["angle"]
            ix_params["order"] = "2"
            ix_params["drag_type"] = "01"
            initial_calibration_data = {
                "cr_params": cr_params,
                "ix_params": ix_params,
                "frequency_offset": 0,
            }
            initial_calibration_data_lst.append(initial_calibration_data)
            
    cr_times = 16 * np.arange(16, duration+16, duration//30, dtype=int)
    return initial_calibration_data_lst, cr_times
# %% Compute the DRAG pulse shape using JAX
def _complex_square_root(x):
    """
    Calculate the complex square root of a given complex number.

    Parameters:
    - x (complex): The input complex number.

    Returns:
    - complex: The complex square root of the input.
    """
    a = jnp.real(x)
    b = jnp.imag(x)
    result = jnp.sign(a) * (
        jnp.sqrt((jnp.abs(x) + a) / 2)
        + 1.0j * jnp.sign(b) * jnp.sqrt((jnp.abs(x) - a) / 2)
    )
    result = jnp.where(
        jnp.abs(b / a) < 1.0e-3,
        # Taylor expansion
        jnp.sign(a)
        * jnp.sqrt(jnp.abs(a))
        * (1 + (1.0j * b / a) / 2 - (1.0j * b / a) ** 2 / 8),
        result,
    )
    result = jnp.where(a == 0.0, 1.0j * b, result)
    return result


def perturbative_pulse_transform_single_photon(pulse_fun, gap, scale=1.0):
    """
    Add perturbative DRAG correction to a pulse based on the pulse function and energy gap.

    Parameters:
    - pulse_fun (callable): The original pulse function.
    - gap (float): The gap parameter for the transformation.
    - scale (float, optional): Scaling factor for the transformation. Default is 1.

    Returns:
    - callable: Transformed pulse function.
    """

    def new_pulse_fun(t, params):
        pulse, dt_pulse = _value_and_jacfwd(pulse_fun)(t, params)
        return pulse - 1.0j * dt_pulse / gap * scale

    return new_pulse_fun


def perturbative_pulse_transform_two_photon(pulse_fun, gap, gaps):
    """
    Add perturbative DRAG correction to a two-photon transition based on the pulse function and energy gaps.

    Parameters:
    - pulse_fun (callable): The original pulse function.
    - gap (float): The common gap parameter for the transformation.
    - gaps (tuple): Tuple containing two gap parameters (gap01, gap12).

    Returns:
    - callable: Transformed pulse function.
    """

    def new_pulse_fun(t, params):
        pulse, dt_pulse = _value_and_jacfwd(pulse_fun)(t, params)
        return _complex_square_root(pulse**2 - 2 * 1.0j * pulse * dt_pulse / gap)

    return new_pulse_fun


def exact_pulse_transform_single_photon(pulse_fun, gap, ratio=1, scale=1.0):
    """
    Apply an exact DRAG correction using Givens rotation to a pulse. It only works for single-photon transitions.

    Parameters:
    - pulse_fun (callable): The original pulse function.
    - gap (float): The gap parameter for the transformation.
    - ratio (float, optional): Ratio parameter. Default is 1.
    - scale (float, optional): Scaling factor for the transformation. Default is 1.

    Returns:
    - callable: Transformed pulse function.
    """

    def new_pulse_fun(t, params):
        pulse = pulse_fun(t, params)
        phase = jnp.where(pulse != 0.0, jnp.exp(1.0j * jnp.angle(pulse)), 0.0)

        def renorm_pulse_fun(t, params):
            pulse = pulse_fun(t, params)
            angle = jnp.angle(pulse)
            theta2 = jnp.arctan(
                -ratio * (params["omega"] / 2) * jnp.abs(pulse) / (gap / 2)
            )
            return theta2, angle

        (theta2, angle), (dt_theta2, dt_angle) = _value_and_jacfwd(renorm_pulse_fun)(
            t, params
        )
        dt_angle = jnp.where(pulse != 0.0, dt_angle, 0.0)
        angle = jnp.where(pulse != 0.0, angle, 0.0)
        new_g = phase * (-(gap + dt_angle) * jnp.tan(theta2) + scale * 1.0j * dt_theta2)
        return jnp.where(
            params["omega"] == 0.0,
            0.0,
            new_g / (ratio * (params["omega"])),
        )

    return new_pulse_fun


def pulse_shape(t, params):
    """
    Define a pulse shape function based on the specified parameters.

    Parameters:
    - t (float): Time parameter.
    - params (dict): Dictionary containing pulse parameters.

    Returns:
    - float: Value of the pulse function at the given time.
    """

    def pulse_fun(t, params):
        order = params["order"]
        if order == "2":
            fun = (
                lambda t, params: t / params["t_r"]
                - (jnp.cos(t * pi / params["t_r"]) * jnp.sin(t * pi / params["t_r"]))
                / pi
            )
        elif order == "1":
            fun = lambda t, params: jnp.sin(jnp.pi / 2 / params["t_r"] * t) ** 2
        elif order == "3":
            t_r = params["t_r"]
            fun = (
                lambda t, params: (jnp.cos(3 * pi * t / t_r)) / 16
                - (9 / 16) * jnp.cos(pi * t / t_r)
                + 1 / 2
            )
        else:
            raise ValueError("Pulse shape order not defined.")
        if "t_r" in params:
            t_r = params["t_r"]
            t_tol = params["t_tol"]
            max_value = fun(params["t_r"], params)
            pulse = max_value
            pulse = jnp.where(t < t_r, fun(t, params), pulse)
            pulse = jnp.where(t > t_tol - t_r, fun(t_tol - t, params), pulse)
            return pulse
        else:
            raise NotImplementedError

    if "drag_scale" not in params or params["drag_scale"] is None:
        drag_scale = jnp.ones(3)
    else:
        drag_scale = jnp.array(params["drag_scale"])

    if "no_drag" in params:
        return pulse_fun(t, params)

    detuning = params["Delta"]
    gap02 = 2 * detuning + params["a1"]
    gap12 = detuning + params["a1"]
    gap01 = detuning

    if "01" in params:
        # Only single-derivative DRAG with gap for level 0 and 1. Use beta from the input parameters as a scaling factor for the DRAG coefficient.
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun, gap01, scale=params.get("beta", 0.0)
        )
    elif "01_scale" in params:
        # Only single-derivative DRAG with gap for level 0 and 1.
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun, gap01, scale=drag_scale[0]
        )
    elif "12" in params:
        # Only single-derivative DRAG with gap for level 1 and 2.
        pulse_fun = perturbative_pulse_transform_single_photon(pulse_fun, gap12)
    elif "02" in params:
        # Only single-derivative DRAG with gap for level 0 and 2.
        pulse_fun = perturbative_pulse_transform_two_photon(
            pulse_fun, gap02, (gap01, gap12)
        )
    elif "exact" in params:
        # Full correction with Givens rotatino for the two single-photon transitions and the perturbative DRAG correction for the two-photon process.
        pulse_fun = perturbative_pulse_transform_two_photon(
            pulse_fun, gap02 / params["drag_scale"][2], (gap01, gap12)
        )
        pulse_fun = exact_pulse_transform_single_photon(
            pulse_fun, gap01 / params["drag_scale"][0], ratio=1
        )
        pulse_fun = exact_pulse_transform_single_photon(
            pulse_fun,
            gap12 / params["drag_scale"][1],
            ratio=jnp.sqrt(2),
            # *(1 - gap12/gap01)
        )
    elif "sw" in params:
        # Full correction for all the three transitions, only use perturabtive DRAG correction.
        pulse_fun = perturbative_pulse_transform_two_photon(
            pulse_fun, gap02 / drag_scale[2], (gap01, gap12)
        )
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun, gap01 / drag_scale[0]
        )
        pulse_fun = perturbative_pulse_transform_single_photon(
            pulse_fun,
            gap12 / drag_scale[1],
        )
    else:
        raise ValueError("Unknown DRAG type.")
    final_pulse = pulse_fun(t, params)
    return final_pulse


# %% Build CR tomography circuit and send jobs


def get_default_cr_params(backend, qc, qt):
    """
    Get the default parameters for the echoed CNOT gate from Qiskit.

    Parameters:
    - backend (qiskit.providers.backend.Backend): The Qiskit backend.
    - qc (int): Control qubit index.
    - qt (int): Target qubit index.

    Returns:
    Tuple[Dict, Dict]: Tuple containing dictionaries of parameters for the echoed CNOT gate for control and target qubits.
    """
    inst_sched_map = backend.instruction_schedule_map

    def _filter_fun(instruction, pulse="ZX"):
        if pulse == "ZX" and "CR90p_u" in instruction[1].pulse.name:
            return True
        elif pulse == "ZX" and "CX_u" in instruction[1].pulse.name:
            return True
        elif pulse == "IX" and "CR90p_d" in instruction[1].pulse.name:
            return True
        elif pulse == "IX" and "CX_d" in instruction[1].pulse.name:
            return True
        return False

    def get_cx_gate_schedule(qc, qt):
        from qiskit.pulse import PulseError

        try:
            return inst_sched_map.get("ecr", (qc, qt))
        except PulseError:
            return inst_sched_map.get("cx", (qc, qt))

    gate_schedule = get_cx_gate_schedule(qc, qt)
    cr_sched_default = gate_schedule.filter(instruction_types=[Play]).filter(
        _filter_fun
    )
    cr_instruction = cr_sched_default.instructions[0][1]
    cr_pulse = cr_instruction.pulse

    ix_sched_default = gate_schedule.filter(instruction_types=[Play]).filter(
        partial(_filter_fun, pulse="IX")
    )
    ix_instruction = ix_sched_default.instructions[0][1]
    ix_channel = ix_instruction.channel
    ix_pulse = ix_instruction.pulse
    return cr_pulse.parameters, ix_pulse.parameters


def _get_modulated_symbolic_pulse(pulse_class, backend, params, frequency_offset):
    """
    Add a detuning to the pulse by pulse shape modulation with exp(2*pi*I*delta*t).

    Parameters:
    - pulse_class (type): The class of the pulse.
    - backend (qiskit.providers.backend.Backend): The Qiskit backend.
    - params (dict): Pulse parameters.
    - frequency_offset (float): Frequency offset for detuning.

    Returns:
    SymbolicPulse: Modulated symbolic pulse.
    """
    default_pulse = pulse_class(**params)
    pulse_parameters = default_pulse.parameters
    pulse_parameters["delta"] = frequency_offset * backend.dt
    _t, _delta = sym.symbols("t, delta")
    pulse_parameters["width"] = float(pulse_parameters["width"])
    pulse_parameters["amp"] = float(pulse_parameters["amp"])
    pulse_parameters["angle"] = float(pulse_parameters["angle"])
    pulse_parameters["delta"] = float(pulse_parameters["delta"])
    my_pulse = SymbolicPulse(
        pulse_type="GaussianModulated",
        duration=int(default_pulse.duration),
        parameters=pulse_parameters,
        envelope=default_pulse.envelope * sym.exp(2 * sym.pi * sym.I * _delta * _t),
        name="modulated Gaussian",
        limit_amplitude=False
    )
    try:
        my_pulse.validate_parameters()
    except:
        raise ValueError("Invalid pulse parameters.")
    return my_pulse


def _add_symbolic_gaussian_pulse(pulse1, pulse2):
    """
    Add two symbolic GaussianSquare pulses.
    It is used in the target drive for the direct CNOT gate calibration to separate the parameter from the echoed CNOT gate.

    Parameters:
    - pulse1 (SymbolicPulse): First symbolic GaussianSquare pulse.
    - pulse2 (SymbolicPulse): Second symbolic GaussianSquare pulse.

    Returns:
    SymbolicPulse: Resulting pulse after adding the two pulses.
    """
    edited_params2 = {key + "_": value for key, value in pulse2.parameters.items()}
    edited_pulse2_envelop = pulse2.envelope
    for p_str in pulse2.parameters.keys():
        old_p = sym.symbols(p_str)
        new_p = sym.symbols(p_str + "_")
        edited_pulse2_envelop = edited_pulse2_envelop.xreplace(old_p, new_p)
    pulse_parameters = pulse1.parameters.copy()
    pulse_parameters.update(edited_params2)
    if pulse1.duration != pulse2.duration:
        raise RuntimeError("Pulse duration must be the same.")
    my_pulse = SymbolicPulse(
        pulse_type="SumGaussian",
        duration=pulse1.duration,
        parameters=pulse_parameters,
        envelope=pulse1.envelope + edited_pulse2_envelop,
        name="added_pulse",
    )
    return my_pulse


def _add_waveform(waveform1, waveform2):
    """Add two qiskit waveform samplings.

    Parameters:
    - waveform1 (Waveform): First qiskit waveform.
    - waveform2 (Waveform): Second qiskit waveform.

    Returns:
    Waveform: Resulting waveform after adding the two waveforms.
    """
    if len(waveform1.samples) != len(waveform2.samples):
        raise ValueError("The two waveforms do not have the same length.")
    return Waveform(
        samples=waveform1.samples + waveform2.samples,
        name=waveform1.name,
        limit_amplitude=False,
    )


def get_custom_pulse_shape(params, backend, control_qubit, frequency_offset=0.0):
    """Compute the custom sine-shaped pulse with DRAG correction for the custom CR gate. It is only possible to get the array sampling of the waveform, no symbolic expressions.

    Parameters:
    - params (dict): Dictionary containing parameters for pulse shape computation.
    - backend: Qiskit backend.
    - control_qubit: Index of the control qubit.
    - frequency_offset (float, optional): Frequency offset. Default is 0.0.

    Returns:
    Waveform: Computed pulse shape with DRAG correction.
    """

    def regulate_array_length(array):
        result_array = np.zeros((len(array) // 16 + 1) * 16)
        result_array[: len(array)] = array
        return result_array

    params = params.copy()
    # Computation of the DRAG pulse needs the final time in ns, transfer "duration" to the final time "t_tol".
    if "duration" in params:
        params["t_tol"] = params["duration"] * backend.dt * 1.0e9
    elif "t_tol" in params:
        pass
    else:
        ValueError(
            "The total time of the CR pulse t_tol is not defined in the parameter dictionary."
        )
    # If amp is given, comute the the drive strength in 2*pi*GHz
    params["omega"] = 2 * pi * amp_to_omega_GHz(backend, control_qubit, params["amp"])
    # Choose the DRAG type
    if "drag_type" in params and params["drag_type"]:
        params[params["drag_type"]] = True
    else:
        params["no_drag"] = True
    del params["drag_type"]

    # Generate time array
    tlist = np.arange(0, params["t_tol"], backend.dt * 1.0e9)
    tlist = regulate_array_length(tlist)

    # Compute pulse shape with DRAG correction using JAX
    pulse_shape_sample = params["amp"] * jax.vmap(pulse_shape, (0, None))(
        tlist,
        params,
    )
    del params["amp"]

    # Conjugate the pulse shape array (because of the qiskit convention) and apply frequency offset
    pulse_shape_sample = np.array(pulse_shape_sample).conjugate()
    pulse_shape_sample = pulse_shape_sample * np.exp(
        2
        * np.pi
        * 1.0j
        * frequency_offset
        * backend.dt
        * np.arange(len(pulse_shape_sample))
    )
    pulse_shape_sample = pulse_shape_sample * np.exp(1.0j * params["angle"])
    # print(pulse_shape_sample.dtype)
    # pulse_shape_sample = jnp.round(pulse_shape_sample.real, 4) + \
        # 1.0j * jnp.round(pulse_shape_sample.imag, 4)
    pulse_shape_sample = np.asarray(pulse_shape_sample, dtype=np.complex64)

    # Return the computed waveform
    return Waveform(
        samples=pulse_shape_sample,
        name="CR drive",
        limit_amplitude=False,
    )

def get_u_channel(backend, qubits):
    back_dict = backend.configuration().to_dict()
    for channel in back_dict['channels']:
        if "u" in channel and "acquire" not in channel:
            channel_dict = back_dict["channels"][channel]
            if channel_dict["operates"]["qubits"] == qubits:
                return(int(channel.replace("u", "")))


def get_cr_schedule(qubits, backend, **kwargs):
    """
    Generate the qiskit Schedule for CR pulse, including both the CR drive and the target.

    Args:
      qubits (Tuple): Tuple of control and target qubits.
      backend (Backend): Qiskit backend.
      **kwargs: Additional keyword arguments:
        - cr_params (dict): Parameters for the CR pulse.
        - ix_params (dict): Parameters for the IX pulse.
        - x_gate_ix_params (dict, optional): Additional parameters for the X-gate on the target IX pulse. Default is None.
        - frequency_offset (float, optional): Frequency offset in . Default is 0.0.

    Returns:
      Schedule: Qiskit Schedule for the CR pulse.
    """
    cr_params = kwargs["cr_params"].copy()
    ix_params = kwargs["ix_params"].copy()
    frequency_offset = kwargs.get("frequency_offset", 0.0)
    x_gate_ix_params = kwargs.get("x_gate_ix_params", None)

    cr_params["risefall_sigma_ratio"] = 2.0
    ix_params["risefall_sigma_ratio"] = 2.0
    if x_gate_ix_params is not None:
        x_gate_ix_params["risefall_sigma_ratio"] = 2.0
    if "width" in cr_params:
        del cr_params["width"]
    if "width" in ix_params:
        del ix_params["width"]
    qc, qt = qubits

    cr_schedule = pulse.Schedule()
    u_id = get_u_channel(backend, [qc, qt])
    control_channel = pulse.ControlChannel(u_id)
    target_channel = pulse.DriveChannel(qt)
    if "beta" not in ix_params:
        ix_params["beta"] = 0.0
    
    # Generate CR pulse waveform
    if "drag_type" in cr_params:
        cr_pulse = get_custom_pulse_shape(
            cr_params, backend, qc, frequency_offset=frequency_offset
        )
    else:
        cr_pulse = _get_modulated_symbolic_pulse(
            GaussianSquare, backend, cr_params, frequency_offset
        )
    
    if "drag_type" in ix_params:
        ix_pulse = get_custom_pulse_shape(
            ix_params, backend, qc, frequency_offset=frequency_offset
        )
        if x_gate_ix_params is not None:
            x_gate_pulse = get_custom_pulse_shape(
                x_gate_ix_params, backend, qc, frequency_offset
            )
            ix_pulse = _add_waveform(ix_pulse, x_gate_pulse)
    else:
        ix_pulse = _get_modulated_symbolic_pulse(
            GaussianSquareDrag, backend, ix_params, frequency_offset
        )
        if x_gate_ix_params is not None:
            x_gate_pulse = _get_modulated_symbolic_pulse(
                GaussianSquare, backend, x_gate_ix_params, frequency_offset
            )
            ix_pulse = _add_symbolic_gaussian_pulse(ix_pulse, x_gate_pulse)
    
    # Check if CR and IX pulse durations are equal
    if cr_pulse.duration != ix_pulse.duration:
        raise RuntimeError(
            f"CR and IX pulse duration are not equal {cr_pulse.duration} and {ix_pulse.duration}."
        )
    # cr_schedule.append(pulse.Play(cr_pulse, control_channel), inplace=True)
    # cr_schedule.append(pulse.Play(ix_pulse, target_channel), inplace=True)

    tmp_circ = QuantumCircuit(backend.num_qubits)
    if backend.name == "DynamicsBackend":
        # pass
        tmp_circ.rz(
            +2 * pi * frequency_offset * backend.dt * ix_pulse.duration,
            qt,
        )
    else:
        tmp_circ.rz(
            -2 * pi * frequency_offset * backend.dt * ix_pulse.duration,
            qt,
        )
    # cr_schedule.append(schedule(tmp_circ, backend=backend), inplace=True)
    with pulse.build(backend=backend) as cr_schedule:
        pulse.play(cr_pulse, control_channel)
        pulse.play(ix_pulse, target_channel)
        pulse.call(schedule(tmp_circ, backend=backend))
    # cr_sched_bk = ScheduleBlock(name="CR schedule")
    # for time, inst in cr_schedule.instructions:
        # cr_sched_bk = cr_sched_bk.append(inst)
    
 
    return cr_schedule

def _generate_indiviual_circuit(
    duration,
    qubits,
    backend,
    cr_params,
    ix_params,
    x_gate_ix_params,
    frequency_offset,
    control_states,
):
    """
    Generate tomography circuits for a given CR pulse duration.

    Args:
      duration (float): Duration of the CR pulse.
      qubits (Tuple): Tuple of control and target qubits.
      backend (Backend): Qiskit backend.
      cr_params (dict): Parameters for the CR pulse.
      ix_params (dict): Parameters for the IX pulse.
      x_gate_ix_params (dict): Parameters for the X-gate on the IX pulse.
      frequency_offset (float): Frequency offset.
      control_states (Tuple): Control states for the tomography circuits.

    Returns:
      List[QuantumCircuit]: List of generated tomography circuits.
    """
    cr_gate = circuit.Gate("cr", num_qubits=2, params=[])
    (qc, qt) = qubits
    circ_list = []
    cr_params["duration"] = int(duration)
    ix_params["duration"] = int(duration)
    if x_gate_ix_params is not None:
        x_gate_ix_params["duration"] = int(duration)

    # Get CR pulse schedule
    cr_sched = get_cr_schedule(
        (qc, qt),
        backend,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
    )

    for basis in ["X", "Y", "Z"]:
        for control in control_states:
            tomo_circ = circuit.QuantumCircuit(
                backend.num_qubits, 2
            )  # use all qubits to avoid error
            if control in (1, "1"):
                tomo_circ.x(qc)  # flip control from |0> to |1>
            tomo_circ.append(cr_gate, [qc, qt])
            tomo_circ.barrier(qc, qt)
            if basis == "X":
                tomo_circ.rz(pi/2, qt)
                tomo_circ.sx(qt)
            elif basis == "Y":
                tomo_circ.sx(qt)
                # tomo_circ.h(qt)
            tomo_circ.measure(qc, 0)
            tomo_circ.measure(qt, 1)
            tomo_circ.add_calibration(gate=cr_gate, qubits=[qc, qt], schedule=cr_sched)
            circ_list.append(tomo_circ)
    return circ_list


def get_cr_tomo_circuits(
    qubits,
    backend,
    cr_times,
    cr_params,
    ix_params,
    x_gate_ix_params=None,
    frequency_offset=0.0,
    control_states=(0, 1),
):
    """
    Build an array of cross resonance schedules for the Hamiltonian tomography experiment.

    Args:
      qubits (Tuple): Tuple of control and target qubits.
      backend (Backend): Qiskit backend.
      cr_times (List[int]): List of CR pulse durations.
      cr_params (dict): Parameters for the CR pulse
      ix_params (dict): Parameters for the IX pulse.
      x_gate_ix_params (dict, optional): Parameters for the X-gate on the IX pulse. Default is None.
      frequency_offset (float, optional): Frequency offset. Default is 0.0.
      control_states (Tuple, optional): Control states for the tomography circuits. Default is (0, 1).

    Returns:
      List[QuantumCircuit]: List of generated tomography circuits.
    """
    tomo_circs = []

    # cr_params = cr_params.copy()
    # ix_params = ix_params.copy()
    if x_gate_ix_params is not None:
        x_gate_ix_params = x_gate_ix_params.copy()
    tomo_circs = []

    tmp_fun = partial(
        _generate_indiviual_circuit,
        qubits=qubits,
        backend=backend,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
        control_states=control_states,
    )

    # with mpl.Pool(10) as p:
    #     tomo_circs = p.map(tmp_fun, cr_times)
    # tomo_circs = map(tmp_fun, cr_times)
    for duration in cr_times:
        tomo_circs.extend(
            _generate_indiviual_circuit(
                duration,
                qubits,
                backend,
                cr_params,
                ix_params,
                x_gate_ix_params,
                frequency_offset,
                control_states,
            )
        )

    # tomo_circs = sum(tomo_circs, [])
    logger.info("Tomography circuits have been generated.")
    return tomo_circs


def _send_decoy_cicuit(backend, session):
    """
    Send a very simple decoy circuit to the backend to prevent idling.

    Parameters:
    - backend (Backend): Qiskit backend.
    - session (Session): Qiskit session.
    """
    qc = QuantumCircuit(1)
    qc.x(0)
    qc = transpile(
        qc,
        session,
        scheduling_method="asap",
        optimization_level=1,
    )
    pub = (qc)
    sampler = Sampler(mode=backend)
    job = sampler.run(
        [pub],
        shots=10,
    )
    return job.job_id()


def send_cr_tomography_job(
    qubits,
    backend,
    cr_params,
    ix_params,
    cr_times,
    x_gate_ix_params=None,
    blocking=False,
    frequency_offset=0.0,
    session=None,
    shots=1024,
    control_states=(0, 1),
    decoy=False,
):
    """
    Send the tomography job for CR pulse.

    Parameters:
    - qubits (Tuple): Tuple of control and target qubits.
    - backend (Backend): Qiskit backend.
    - cr_params (dict): Parameters for the CR pulse.
    - ix_params (dict): Parameters for the IX pulse.
    - cr_times (List[float]): List of widths of the CR pulses.
    - x_gate_ix_params (dict, optional): Parameters for the X-gate IX pulse. Default is None.
    - blocking (bool, optional): If True, block until the job is completed. Default is False.
    - frequency_offset (float, optional): Frequency offset. Default is 0.0.
    - ZI_correction (float, optional): ZI interaction rate correction. Default is 0.0.
    - session (Session, optional): Qiskit session. Default is None.
    - shots (int, optional): Number of shots. Default is 1024.
    - control_states (Tuple, optional): Tuple of control states. Default is (0, 1).
    - decoy (bool, optional): If True, send a decoy circuit to prevent idling. Default is True.

    Returns:
    str: Job ID.
    """
    # Create circuits
    cr_tomo_circ_list = get_cr_tomo_circuits(
        qubits,
        backend,
        cr_times,
        cr_params=cr_params,
        ix_params=ix_params,
        x_gate_ix_params=x_gate_ix_params,
        frequency_offset=frequency_offset,
        control_states=control_states,
    )
    if decoy:
        _send_decoy_cicuit(backend, session)

    transpiled_tomo_circ_list = transpile(
        cr_tomo_circ_list,
        backend,
        # scheduling_method="asap",
        optimization_level=1,
    )
    # Send jobs
    if session is not None:
        sampler = Sampler(mode=backend)
        pub_circ = [(circ) for circ in transpiled_tomo_circ_list]
        job = sampler.run(
            pub_circ,
        )
    else:
        job = backend.run(transpiled_tomo_circ_list, shots=shots)
    tag = "CR tomography"
    # job.update_tags([tag])  # qiskit-ibm-runtime does not support yet update_tags, will support soon.
    parameters = {
        "backend": backend.name,
        "qubits": qubits,
        "cr_times": cr_times,
        "shots": shots,
        "cr_params": cr_params,
        "ix_params": ix_params,
        "x_gate_ix_params": x_gate_ix_params,
        "frequency_offset": frequency_offset,
        "dt": backend.dt,
    }
    logger.info(
        tag
        + ": "
        + job.job_id()
        + "\n"
        + "\n".join([f"{key}: {val}" for key, val in parameters.items()])
        + "\n"
    )
    if blocking:
        save_job_data(job, backend=backend, parameters=parameters)
    else:
        async_execute(save_job_data, job, backend=backend, parameters=parameters)
    return job.job_id()


def shifted_parallel_parameter_cr_job(
        qubit_pairs,
        backend,
        cr_params_lst,
        ix_params_lst,
        cr_times,
        prob_ix_strength_lst,
        x_gate_ix_params_lst=None,
        frequency_offset_lst=None,
        shots=1024,
        blocking=False,
        control_states=(0, 1),
        mode="CR",
):
    if mode == "CR":
        ix_params_lst = deepcopy(ix_params_lst)
        for index, ix_params in enumerate(ix_params_lst):
            ix_params["amp"] += prob_ix_strength_lst[index]
    else:
        x_gate_ix_params_lst = deepcopy(x_gate_ix_params_lst)
        # print("x_gate_ix_params_lst line 853:", x_gate_ix_params_lst)
        # print("prob_ix_strength_lst line 854:", prob_ix_strength_lst)
        for index, x_gate_ix_params in enumerate(x_gate_ix_params_lst):
            x_gate_ix_params["amp"] += prob_ix_strength_lst[index]
    # print("cr_pulse line 850:", x_gate_ix_params_lst)
    tomo_id = parallel_send_cr_tomography_job(
        qubit_pairs,
        backend,
        cr_params_lst,
        ix_params_lst,
        cr_times,
        x_gate_ix_params_lst=x_gate_ix_params_lst,
        frequency_offset_lst=frequency_offset_lst,
        shots=shots,
        blocking=blocking,
        control_states=control_states,
        # mode=mode,
    )
    return tomo_id


def shifted_parameter_cr_job(
    qubits,
    backend,
    cr_params,
    ix_params,
    cr_times,
    prob_ix_strength,
    x_gate_ix_params=None,
    frequency_offset=0.0,
    shots=1024,
    blocking=False,
    session=None,
    control_states=(0, 1),
    mode="CR",
):
    """
    Send the tomography job for CR pulse with a shifted amplitude on the target drive.

    Parameters:
    - qubits (Tuple): Tuple of control and target qubits.
    - backend (Backend): Qiskit backend.
    - cr_params (dict): Parameters for the CR pulse.
    - ix_params (dict): Parameters for the IX pulse.
    - cr_times (List[float]): List of widths of the CR pulses.
    - prob_ix_strength (float): Strength of the probability amplitude shift.
    - x_gate_ix_params (dict, optional): Parameters for the X-gate IX pulse. Default is None.
    - frequency_offset (float, optional): Frequency offset. Default is 0.0.
    - shots (int, optional): Number of shots. Default is 1024.
    - blocking (bool, optional): If True, block until the job is completed. Default is False.
    - session (Session, optional): Qiskit session. Default is None.
    - control_states (Tuple, optional): Tuple of control states. Default is (0, 1).
    - mode (str, optional): Operation mode ("CR" or "IX"). Default is "CR".

    Returns:
    str: Job ID.
    """
    if mode == "CR":
        # ix_params = ix_params.copy()
        ix_params = deepcopy(ix_params)
        ix_params["amp"] += prob_ix_strength
    else:
        # x_gate_ix_params = x_gate_ix_params.copy()
        x_gate_ix_params = deepcopy(x_gate_ix_params)
        x_gate_ix_params["amp"] += prob_ix_strength
    tomo_id = send_cr_tomography_job(
        qubits,
        backend,
        cr_params,
        ix_params,
        cr_times,
        x_gate_ix_params,
        blocking=blocking,
        session=session,
        frequency_offset=frequency_offset,
        shots=shots,
        control_states=control_states,
    )
    return tomo_id


# %% Analyze CR tomography data
# Part of the following code is from https://github.com/Qiskit/textbook/blob/main/notebooks/quantum-hardware-pulses/hamiltonian-tomography.ipynb
def _get_omega(eDelta, eOmega_x, eOmega_y):
    r"""Return \Omega from parameter arguments."""
    eOmega = np.sqrt(eDelta**2 + eOmega_x**2 + eOmega_y**2)
    return eOmega


def _avg_X(t, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=True):
    """Return average X Pauli measurement vs time t"""
    if normalize:
        eOmega = _get_omega(eDelta, eOmega_x, eOmega_y)
    eXt = (
        -eDelta * eOmega_x
        + eDelta * eOmega_x * np.cos(eOmega * t + t0)
        + eOmega * eOmega_y * np.sin(eOmega * t + t0)
    ) / eOmega**2
    return eXt


def _avg_Y(t, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=True):
    """Return average Y Pauli measurement vs time t"""
    if normalize:
        eOmega = _get_omega(eDelta, eOmega_x, eOmega_y)
    eYt = (
        eDelta * eOmega_y
        - eDelta * eOmega_y * np.cos(eOmega * t + t0)
        - eOmega * eOmega_x * np.sin(eOmega * t + t0)
    ) / eOmega**2
    return eYt


def _avg_Z(t, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=True):
    """Return average Z Pauli measurement vs time t"""
    if normalize:
        eOmega = _get_omega(eDelta, eOmega_x, eOmega_y)
    eZt = (
        eDelta**2 + (eOmega_x**2 + eOmega_y**2) * np.cos(eOmega * t + t0)
    ) / eOmega**2
    return eZt


def fit_evolution(tlist, eXt, eYt, eZt, p0, include, normalize):
    """
    Use curve_fit to determine fit parameters of X,Y,Z Pauli measurements together.

    Parameters:
    - tlist (array-like): Time values for the measurements.
    - eXt (array-like): X Pauli measurements.
    - eYt (array-like): Y Pauli measurements.
    - eZt (array-like): Z Pauli measurements.
    - p0 (array-like): Initial guess for fit parameters.
    - include (array-like): Boolean array specifying which Pauli measurements to include in the fit.
    - normalize (bool): Whether to normalize the measurements.

    Returns:
    Tuple[array-like, array-like]: Fit parameters and covariance matrix.
    """

    def fun(tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0):
        """Stack average X,Y,Z Pauli measurements vertically."""
        result = np.vstack(
            [
                _avg_X(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Y(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Z(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
            ]
        )[include].flatten()
        return result

    data = np.asarray(
        [
            eXt,
            eYt,
            eZt,
        ]
    )[include].flatten()
    # params, cov = curve_fit(fun, tlist, data, p0=p0, method="trf", maxfev=10000)
    params, cov = curve_fit(fun, tlist, data, p0=p0, maxfev=10000)
    return params, cov


def fit_rt_evol(tlist, eXt, eYt, eZt, p0, include, normalize):
    """
    Use curve_fit to determine fit parameters of X,Y,Z Pauli measurements together.

    Parameters:
    - tlist (array-like): Time values for the measurements.
    - eXt (array-like): X Pauli measurements.
    - eYt (array-like): Y Pauli measurements.
    - eZt (array-like): Z Pauli measurements.
    - p0 (array-like): Initial guess for fit parameters.
    - include (array-like): Boolean array specifying which Pauli measurements to include in the fit.
    - normalize (bool): Whether to normalize the measurements.

    Returns:
    Tuple[array-like, array-like]: Fit parameters and covariance matrix.
    """

    def fun(tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0):
        """
        Stack average X,Y,Z Pauli measurements vertically.
        """
        result = np.vstack(
            [
                _avg_X(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Y(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
                _avg_Z(
                    tlist, eDelta, eOmega_x, eOmega_y, eOmega, t0, normalize=normalize
                ),
            ]
        )[include].flatten()
        return result

    data = np.asarray(
        [
            eXt,
            eYt,
            eZt,
        ]
    )[include].flatten()
    params, cov = curve_fit(fun, tlist, data, p0=p0, method="trf")
    return params, cov


def recursive_fit(tlist, eXt, eYt, eZt, p0):
    """
    Perform recursive fitting of X, Y, Z Pauli expectation values.

    Parameters:
    - tlist (array-like): Time values for the measurements.
    - eXt (array-like): X Pauli expectation values.
    - eYt (array-like): Y Pauli expectation values.
    - eZt (array-like): Z Pauli expectation values.
    - p0 (array-like): Initial guess for fit parameters.

    Returns:
    Tuple[array-like, array-like]: Fit parameters and covariance matrix.
    """
    params = p0.copy()
    # First fit with Z measurement only, no normalization
    include = np.array([False, False, True])
    try:
        params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=False)
    except:
        pass

    # Second fit with Z measurement only, normalization applied
    include = np.array([False, False, True])
    try:
        params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=True)
    except:
        pass

    # Third fit with Y and Z measurements, no normalization
    include = np.array([False, True, True])
    try:
        params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=False)
    except:
        pass
    try:
        params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=True)
    except:
        pass

    # Fourth fit with X, Y, and Z measurements, no normalization
    include = np.array([True, True, True])
    try:
        params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=False)
    except:
        pass

    # Fifth fit with X, Y, and Z measurements, normalization applied
    include = np.array([True, True, True])
    try:
        params, cov = fit_evolution(tlist, eXt, eYt, eZt, params, include, normalize=True)
    except:
        pass

    return params, cov

from scipy.optimize import curve_fit

def get_omega(eDelta, eOmega_x, eOmega_y):
    """Return \Omega from parameter arguments."""
    eOmega = np.sqrt(eDelta**2 + eOmega_x**2 + eOmega_y**2)
    return eOmega

def avg_X(t, eDelta, eOmega_x, eOmega_y, b, t_off):
    """Return average X Pauli measurement vs time t"""
    eOmega = get_omega(eDelta, eOmega_x, eOmega_y)
    eXt = (-eDelta*eOmega_x + eDelta*eOmega_x*np.cos(eOmega*(t+t_off)) + \
           eOmega*eOmega_y*np.sin(eOmega*(t+t_off))) / eOmega**2 + b
    return eXt

def avg_Y(t, eDelta, eOmega_x, eOmega_y, b, t_off):
    """Return average Y Pauli measurement vs time t"""
    eOmega = get_omega(eDelta, eOmega_x, eOmega_y)
    eYt = (eDelta*eOmega_y - eDelta*eOmega_y*np.cos(eOmega*(t+t_off)) - \
           eOmega*eOmega_x*np.sin(eOmega*(t+t_off))) / eOmega**2 + b
    return eYt

def avg_Z(t, eDelta, eOmega_x, eOmega_y, b, t_off):
    """Return average Z Pauli measurement vs time t"""
    eOmega = get_omega(eDelta, eOmega_x, eOmega_y)
    eZt = (eDelta**2 + (eOmega_x**2 + eOmega_y**2)*np.cos(eOmega*(t+t_off))) / eOmega**2\
            + b
    return eZt

def rt_evol(ts, eDelta, eOmega_x, eOmega_y, b, t_off):
    """Stack average X,Y,Z Pauli measurements vertically."""
    return np.vstack([avg_X(ts, eDelta, eOmega_x, eOmega_y, b, t_off), \
                     avg_Y(ts, eDelta, eOmega_x, eOmega_y, b, t_off), \
                     avg_Z(ts, eDelta, eOmega_x, eOmega_y, b, t_off)])
    
def rt_flat(ts, eDelta, eOmega_x, eOmega_y, b, t_off):
    """Flatten X,Y,Z Pauli measurement data into 1D array."""
    return rt_evol(ts[0:len(ts)//3], eDelta, eOmega_x, eOmega_y, b, t_off).flatten()

def fit_rt_evol(ts, eXt, eYt, eZt, p0):
    """Use curve_fit to determine fit parameters of X,Y,Z Pauli measurements together."""
    rt_vec = np.asarray([eXt, eYt, eZt])
    
    return curve_fit(rt_flat, np.tile(ts, 3), rt_vec.flatten(), p0=p0, 
                     method='trf', 
                     bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]))

def simple_fit_single(tlist, eXt, eYt, eZt):
    try: 
        ground_params1, ground_cov1 = fit_rt_evol(tlist, eXt, eYt, eZt, p0=[0.0002, 0.002, 0.001, 0, 0])
    except: 
        ground_params1, ground_cov1, ground_error1 = (None, None, np.inf)
    try:
        ground_params2, ground_cov2 = fit_rt_evol(tlist, eXt, eYt, eZt, p0=[0.0002, 0.0002, 0.0005, 0, 0])
    except:
        ground_params2, ground_cov2, ground_error2 = (None, None, np.inf)
    try:
        ground_params3, ground_cov3 = fit_rt_evol(tlist, eXt, eYt, eZt, p0=[0.00002, 0.00002, 0.00005, 0, 0])
    except:
        ground_params3, ground_cov3, ground_error3 = (None, None, np.inf)
    
    if type(ground_cov1) != type(None):
        ground_error1 = sum(np.sqrt(np.diag(ground_cov1)))
    if type(ground_cov2) != type(None):
        ground_error2 = sum(np.sqrt(np.diag(ground_cov2)))
    if type(ground_cov3) != type(None):
        ground_error3 = sum(np.sqrt(np.diag(ground_cov3)))
    
    ground_error_lst = [ground_error1, ground_error2, ground_error3]
    ground_params_lst = [ground_params1, ground_params2, ground_params3]
    ground_cov_lst = [ground_cov1, ground_cov2, ground_cov3]
    minimal_index = ground_error_lst.index(min(ground_error_lst))
    ground_params, ground_cov = ground_params_lst[minimal_index], ground_cov_lst[minimal_index]
    # ground_params, ground_cov = (ground_params1, ground_cov1) if ground_error1 < ground_error2 else (ground_params2, ground_cov2)

    return ground_params, ground_cov

def simple_fit(tlist, eXt, eYt, eZt):
    '''
    Simple fit to determine the interaction rates.
    '''
    try:
        ground_params1, ground_cov1 = fit_rt_evol(tlist, eXt[0], eYt[0], eZt[0], p0=[0.0002, 0.002, 0.001, 0, 0]) #0.0002, 0.002, 0.001, 0, 0
    except:
        ground_params1, ground_cov1, ground_error1 = (None, None, np.inf)
    try:
        ground_params2, ground_cov2 = fit_rt_evol(tlist, eXt[0], eYt[0], eZt[0], p0=[0.0002, 0.0002, 0.0005, 0, 0]) #0.0002, 0.0002, 0.0005, 0, 0
    except:
        ground_params2, ground_cov2, ground_error2 = (None, None, np.inf)
    try:
        ground_params3, ground_cov3 = fit_rt_evol(tlist, eXt[0], eYt[0], eZt[0], p0=[0.00002, 0.00002, 0.00005, 0, 0]) #0.0002, 0.0002, 0.0005, 0, 0
    except:
        ground_params3, ground_cov3, ground_error3 = (None, None, np.inf)

    if type(ground_cov1) != type(None):
        ground_error1 = sum(np.sqrt(np.diag(ground_cov1)))
    if type(ground_cov2) != type(None):
        ground_error2 = sum(np.sqrt(np.diag(ground_cov2)))
    if type(ground_cov3) != type(None):
        ground_error3 = sum(np.sqrt(np.diag(ground_cov3)))
    ground_error_lst = [ground_error1, ground_error2, ground_error3]
    ground_params_lst = [ground_params1, ground_params2, ground_params3]
    ground_cov_lst = [ground_cov1, ground_cov2, ground_cov3]
    # ground_params, ground_cov = (ground_params1, ground_cov1) if ground_error1 < ground_error2 else (ground_params2, ground_cov2)
    minimal_index = ground_error_lst.index(min(ground_error_lst))
    ground_params, ground_cov = ground_params_lst[minimal_index], ground_cov_lst[minimal_index]

    try:
        excited_params1, excited_cov1 = fit_rt_evol(tlist, eXt[1], eYt[1], eZt[1], p0=[0.0002, 0.001, 0.001, 0, 0])
    except:
        excited_params1, excited_cov1, excited_error1 = (None, None, np.inf)
    try:    
        excited_params2, excited_cov2 = fit_rt_evol(tlist, eXt[1], eYt[1], eZt[1], p0=[0.0002, 0.0002, 0.0005, 0, 0])
    except:
        excited_params2, excited_cov2, excited_error2 = (None, None, np.inf)
    try:
        excited_params3, excited_cov3 = fit_rt_evol(tlist, eXt[1], eYt[1], eZt[1], p0=[0.00002, 0.00002, 0.00005, 0, 0])
    except:
        excited_params3, excited_cov3, excited_error3 = (None, None, np.inf)
    
    if type(excited_cov1) != type(None):
        excited_error1 = sum(np.sqrt(np.diag(excited_cov1)))
    if type(excited_cov2) != type(None):
        excited_error2 = sum(np.sqrt(np.diag(excited_cov2)))
    if type(excited_cov3) != type(None):
        excited_error3 = sum(np.sqrt(np.diag(excited_cov3)))
    # excited_params, excited_cov = (excited_params1, excited_cov1) if excited_error1 < excited_error2 else (excited_params2, excited_cov2)
    excited_error_lst = [excited_error1, excited_error2, excited_error3]
    excited_params_lst = [excited_params1, excited_params2, excited_params3]
    excited_cov_lst = [excited_cov1, excited_cov2, excited_cov3]
    minimal_index = excited_error_lst.index(min(excited_error_lst))
    excited_params, excited_cov = excited_params_lst[minimal_index], excited_cov_lst[minimal_index]
    return ground_params, ground_cov, excited_params, excited_cov


def get_interation_rates_MHz_simple(ground_fit, excited_fit, ground_cov, excited_cov, dt):
    """
    Determine interaction rates from fits to ground and excited control qubit data.
    """
    Delta0 = (ground_fit[0]/dt)/1e6
    Omega0_x = (ground_fit[1]/dt)/1e6
    Omega0_y = (ground_fit[2]/dt)/1e6
    Delta1 = (excited_fit[0]/dt)/1e6
    Omega1_x = (excited_fit[1]/dt)/1e6
    Omega1_y = (excited_fit[2]/dt)/1e6
    Delta0_var, Omega0_x_var, Omega0_y_var = np.diag(ground_cov)[:3]
    Delta1_var, Omega1_x_var, Omega1_y_var = np.diag(excited_cov)[:3]
    IX_std = 0.5 * (Omega0_x_var + Omega1_x_var) ** 0.5 / 2 / pi
    IY_std = 0.5 * (Omega0_y_var + Omega1_y_var) ** 0.5 / 2 / pi
    IZ_std = 0.5 * (Delta0_var + Delta1_var) ** 0.5 / 2 / pi
    ZX_std = 0.5 * (Omega0_x_var + Omega1_x_var) ** 0.5 / 2 / pi
    ZY_std = 0.5 * (Omega0_y_var + Omega1_y_var) ** 0.5 / 2 / pi
    ZZ_std = 0.5 * (Delta0_var + Delta1_var) ** 0.5 / 2 / pi
    
    IX = 0.5*(Omega0_x + Omega1_x)
    IY = 0.5*(Omega0_y + Omega1_y)
    IZ = 0.5*(Delta0 + Delta1)
    ZX = 0.5*(Omega0_x - Omega1_x)
    ZY = 0.5*(Omega0_y - Omega1_y)
    ZZ = 0.5*(Delta0 - Delta1)
    
    return [[IX, IY, IZ], [ZX, ZY, ZZ]], [
        [IX_std, IY_std, IZ_std],
        [ZX_std, ZY_std, ZZ_std],
    ]

def get_interation_rates_MHz(ground_params, excited_params, ground_cov, excited_cov):
    """
    Determine two-qubits interaction rates from fits to ground and excited control qubit data.

    Parameters:
    - ground_params (array-like): Fit parameters for the ground state.
    - excited_params (array-like): Fit parameters for the excited state.
    - ground_cov (array-like): Covariance matrix for the ground state fit.
    - excited_cov (array-like): Covariance matrix for the excited state fit.

    Returns:
    Tuple[array-like, array-like]: Interaction rates and their standard deviations.
    """
    Delta0, Omega0_x, Omega0_y = ground_params[:3]
    Delta1, Omega1_x, Omega1_y = excited_params[:3]
    Delta0_var, Omega0_x_var, Omega0_y_var = np.diag(ground_cov)[:3]
    Delta1_var, Omega1_x_var, Omega1_y_var = np.diag(excited_cov)[:3]

    # Interaction rates
    IX = 0.5 * (Omega0_x + Omega1_x) / 2 / pi
    IY = 0.5 * (Omega0_y + Omega1_y) / 2 / pi
    IZ = 0.5 * (Delta0 + Delta1) / 2 / pi
    ZX = 0.5 * (Omega0_x - Omega1_x) / 2 / pi
    ZY = 0.5 * (Omega0_y - Omega1_y) / 2 / pi
    ZZ = 0.5 * (Delta0 - Delta1) / 2 / pi

    # Standard deviations
    IX_std = 0.5 * (Omega0_x_var + Omega1_x_var) ** 0.5 / 2 / pi
    IY_std = 0.5 * (Omega0_y_var + Omega1_y_var) ** 0.5 / 2 / pi
    IZ_std = 0.5 * (Delta0_var + Delta1_var) ** 0.5 / 2 / pi
    ZX_std = 0.5 * (Omega0_x_var + Omega1_x_var) ** 0.5 / 2 / pi
    ZY_std = 0.5 * (Omega0_y_var + Omega1_y_var) ** 0.5 / 2 / pi
    ZZ_std = 0.5 * (Delta0_var + Delta1_var) ** 0.5 / 2 / pi

    return [[IX, IY, IZ], [ZX, ZY, ZZ]], [
        [IX_std, IY_std, IZ_std],
        [ZX_std, ZY_std, ZZ_std],
    ]


def _estimate_period(data, cr_times):
    """
    Estimate the period of the oscillatory data using peak finding.

    Parameters:
    - data (array-like): Oscillatory data.
    - cr_times (array-like): Corresponding time values.

    Returns:
    float: Estimated period of the oscillatory data.
    """
    peaks_high, properties = scipy.signal.find_peaks(data, prominence=0.5)
    peaks_low, properties = scipy.signal.find_peaks(-data, prominence=0.5)
    peaks = sorted(np.concatenate([peaks_low, peaks_high]))
    if len(peaks) <= 2:
        return cr_times[-1] - cr_times[0]
    return 2 * np.mean(np.diff(cr_times[peaks]))


def _get_normalized_cr_tomography_data(job_id, parallel_exp=False,
                                       qubit_pair=None):
    """Retrieve and normalize CR tomography data from a job. Renormalize the data to (-1, 1).

    Args:
        job_id (str): The ID of the job containing CR tomography data.

    Returns:
        Tuple[array-like, array-like]: A tuple containing the CR times and normalized tomography data.
    """
    data = load_job_data(job_id)
    result = data["result"]
    dt = data["parameters"]["dt"]
    cr_times = data["parameters"]["cr_times"]
    shots = data["parameters"]["shots"]

    # IBM classified data
    # Trace out the control, notice that IBM uses reverse qubit indices labeling.
    if parallel_exp == False:
        target_data = (
            np.array(
                [
                    (result.get_counts(i).get("00", 0) + result.get_counts(i).get("01", 0))
                    for i in range(len(result.results))
                ]
            )
            / shots
        )
    else:
        target_data = np.zeros(len(result.results))
        qubit_pairs = data["parameters"]["qubit_pairs"]
        qubit_pair_index = qubit_pairs.index(qubit_pair)
        target_id = len(qubit_pairs) * 2 - 2 * qubit_pair_index - 2
        for i in range(len(result.results)):
           for _ in result.get_counts(i):
               if _[target_id] == "0":
                    target_data[i] += result.get_counts(i)[_] / shots
                   
    # print(target_data)

    if 6 * len(cr_times) == len(target_data):
        # two-qubit tomography, with control on 0 and 1
        splitted_data = target_data.reshape((len(cr_times), 6)).transpose()
    elif 3 * len(cr_times) == len(target_data):
        # single tomography
        splitted_data = target_data.reshape((len(cr_times), 3)).transpose()
    else:
        ValueError(
            "The number of data points does not match the number of tomography settings."
        )
    splitted_data = splitted_data * 2 - 1

    scale = np.max(splitted_data) - np.min(splitted_data)
    average = (np.max(splitted_data) + np.min(splitted_data)) / 2
    splitted_data = 2 * (splitted_data - average) / scale

    return cr_times, splitted_data, dt


def get_interact_single(cr_times, splitted_data, dt, show_plot=False):
    signal_x = splitted_data[0]
    signal_y = splitted_data[1]
    signal_z = splitted_data[2]
    period = _estimate_period(signal_z, cr_times)
    cutoff = -1
    params, cov = recursive_fit(
        cr_times[:cutoff] * dt * 1.0e6,
        signal_x[:cutoff],
        signal_y[:cutoff],
        signal_z[:cutoff],
        p0=np.array([1, 1, 1, 1 / (period * dt * 1.0e6) * 2 * pi, 0.0]),
    )
    ground_params, ground_cov = simple_fit_single(
        cr_times[:cutoff],
        signal_x[:cutoff],
        signal_y[:cutoff],
        signal_z[:cutoff],
    )
    error1 = np.sum((signal_x - _avg_X(cr_times*dt*1e6, *params)) **2) + \
             np.sum((signal_y - _avg_Y(cr_times*dt*1e6, *params)) **2) + \
             np.sum((signal_z - _avg_Z(cr_times*dt*1e6, *params)) **2)
    error2 = np.sum((signal_x - avg_X(cr_times, *ground_params))** 2) + \
                np.sum((signal_y - avg_Y(cr_times, *ground_params))** 2) + \
                np.sum((signal_z - avg_Z(cr_times, *ground_params))** 2)
    if error1 < error2:
        if show_plot:
            plot_cr_ham_tomo(
                cr_times * dt * 1.0e6,
                splitted_data,
                ground_params=params,
                ground_cov=cov,
            )
            plt.show()
        return {
        "IX": params[1] / 2 / pi,
        "IY": params[2] / 2 / pi,
        "IZ": params[0] / 2 / pi,
    }

    else:
        if show_plot:
            plot_cr_ham_tomo_simple_single(
                cr_times,
                splitted_data,
                ground_params=ground_params,
                ground_cov=ground_cov,
                dt=dt,
            )
            plt.show()
        return {
        "IX": ground_params[1] / dt / 1e6 / 2 / pi,
        "IY": ground_params[2] / dt/ 1e6 / 2 / pi,
        "IZ": ground_params[0] / dt/ 1e6/ 2 / pi,
    }


def get_interact(cr_times, splitted_data, dt, show_plot=False):
    signal_x = splitted_data[:2]
    signal_y = splitted_data[2:4]
    signal_z = splitted_data[4:6]

    period0 = _estimate_period(signal_z[0], cr_times)
    period1 = _estimate_period(signal_z[1], cr_times)

    cutoff = -1
    _i = 0
    while True:
        try:
            ground_params1, ground_cov1 = recursive_fit(
                cr_times[:cutoff] * dt * 1.0e6,
                signal_x[0][:cutoff],
                signal_y[0][:cutoff],
                signal_z[0][:cutoff],
                p0=np.array([1, 1, 1, 1 / (period0 * dt * 1.0e6) * 2 * pi, 0.0]),
            )
            break
        except RuntimeError as e:
            _i += 1
            period0 *= 2
            if _i > 16:
                raise e
    ground_params2, ground_cov2, excited_params2, excited_cov2 = simple_fit(
        cr_times,
        signal_x,
        signal_y,
        signal_z,
    )
            

    excited_params1, excited_cov1 = recursive_fit(
        cr_times[:cutoff] * dt * 1.0e6,
        signal_x[1][:cutoff],
        signal_y[1][:cutoff],
        signal_z[1][:cutoff],
        p0=np.array([1, 1, 1, 1 / (period1 * dt * 1.0e6) * 2 * pi, 0.0]),
    )
    # ground_params, ground_cov = excited_params, excited_cov
    error1 = np.sum((signal_x[0] - _avg_X(cr_times*dt*1e6, *ground_params1)) **2) + \
             np.sum((signal_y[0] - _avg_Y(cr_times*dt*1e6, *ground_params1)) **2) + \
             np.sum((signal_z[0] - _avg_Z(cr_times*dt*1e6, *ground_params1)) **2) + \
             np.sum((signal_x[1] - _avg_X(cr_times*dt*1e6, *excited_params1))**2) + \
             np.sum((signal_y[1] - _avg_Y(cr_times*dt*1e6, *excited_params1))**2) + \
             np.sum((signal_z[1] - _avg_Z(cr_times*dt*1e6, *excited_params1))**2)
    if excited_params2 is None or ground_params2 is None:
        error2 = np.inf
    else:
        error2 = np.sum((signal_x[0] - avg_X(cr_times, *ground_params2))** 2) + \
                 np.sum((signal_y[0] - avg_Y(cr_times, *ground_params2))** 2) + \
                 np.sum((signal_z[0] - avg_Z(cr_times, *ground_params2))** 2) + \
                 np.sum((signal_x[1] - avg_X(cr_times, *excited_params2))**2) + \
                 np.sum((signal_y[1] - avg_Y(cr_times, *excited_params2))**2) + \
                 np.sum((signal_z[1] - avg_Z(cr_times, *excited_params2))**2)
    if error1 < error2:
        if show_plot:
            plot_cr_ham_tomo(
                cr_times * dt * 1.0e6,
                splitted_data,
                ground_params=ground_params1,
                excited_params=excited_params1,
                ground_cov=ground_cov1,
                excited_cov=excited_cov1,
            )
            plt.show()
        [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz(
            ground_params1, excited_params1, ground_cov1, excited_cov1
        )[0]
    else:
        if show_plot:
            plot_cr_ham_tomo_simple(
                cr_times,
                splitted_data,
                ground_params=ground_params2,
                excited_params=excited_params2,
                ground_cov=ground_cov2,
                excited_cov=excited_cov2,
                dt=dt,
            )
            plt.show()
        [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz_simple(
            ground_params2, excited_params2, ground_cov2, excited_cov2, dt=dt
        )[0]

    # if show_plot:
    #     plot_cr_ham_tomo_simple(
    #         cr_times,
    #         splitted_data,
    #         ground_params,
    #         excited_params,
    #         ground_cov,
    #         excited_cov,
    #         dt=dt,
    #     )
    #     plt.show()
    # [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz_simple(
    #     ground_pa rams, excited_params, ground_cov, excited_cov, dt=dt
    # )[0]
    return {"IX": IX, "IY": IY, "IZ": IZ, "ZX": ZX, "ZY": ZY, "ZZ": ZZ}
   

    
def process_single_qubit_tomo_data(job_id, show_plot=False):
    """Process and analyze single qubit tomography data from a job.

    Args:
        job_id (str): The ID of the job containing single qubit \
tomography data.
        show_plot (bool, optional): Whether to generate and \
display plots. Default is False.

    Returns:
        dict: Dictionary containing the processed results including <X>, \
<Y>, and <Z>.

    Note:
        Noticed that the measured value is not the IX, IY, IZ in the sense of \
CR, but the single qubit dynamics when the control qubit is in |0> or |1>.
    """
    cr_times, splitted_data, dt = _get_normalized_cr_tomography_data(job_id)
    signal_x, signal_y, signal_z = splitted_data

    period = _estimate_period(signal_z, cr_times)
    cutoff = -1
    params, cov = recursive_fit(
        cr_times[:cutoff] * dt * 1.0e6,
        signal_x[:cutoff],
        signal_y[:cutoff],
        signal_z[:cutoff],
        p0=np.array([1, 1, 1, 1 / (period * dt * 1.0e6) * 2 * pi, 0.0]),
    )
    ground_params, ground_cov = simple_fit_single(
        cr_times[:cutoff],
        signal_x[:cutoff],
        signal_y[:cutoff],
        signal_z[:cutoff],
    )
    error1 = np.sum((signal_x - _avg_X(cr_times*dt*1e6, *params)) **2) + \
             np.sum((signal_y - _avg_Y(cr_times*dt*1e6, *params)) **2) + \
             np.sum((signal_z - _avg_Z(cr_times*dt*1e6, *params)) **2)
    error2 = np.sum((signal_x - avg_X(cr_times, *ground_params))** 2) + \
                np.sum((signal_y - avg_Y(cr_times, *ground_params))** 2) + \
                np.sum((signal_z - avg_Z(cr_times, *ground_params))** 2)
    if error1 < error2:
        if show_plot:
            plot_cr_ham_tomo(
                cr_times * dt * 1.0e6,
                splitted_data,
                ground_params=params,
                ground_cov=cov,
            )
            plt.show()
        return {
        "IX": params[1] / 2 / pi,
        "IY": params[2] / 2 / pi,
        "IZ": params[0] / 2 / pi,
    }

    else:
        if show_plot:
            plot_cr_ham_tomo_simple_single(
                cr_times,
                splitted_data,
                ground_params=ground_params,
                ground_cov=ground_cov,
                dt=dt,
            )
            plt.show()
        return {
        "IX": ground_params[1] / dt / 1e6 / 2 / pi,
        "IY": ground_params[2] / dt/ 1e6 / 2 / pi,
        "IZ": ground_params[0] / dt/ 1e6/ 2 / pi,
    }
    

## TODO remove dt in the signiture.
def process_zx_tomo_data(job_id, show_plot=False,
                         parallel_exp=False, qubit_pair=None):
    """Process and analyze ZX tomography data from a job.

    Args:
        job_id (str): The ID of the job containing ZX tomography data.
        show_plot (bool, optional): Whether to generate and display plots. Default is False.

    Returns:
        dict: Dictionary containing the processed results including IX, IY, IZ, ZX, ZY, and ZZ.

    Note:
        The effective coupling strength is in the unit of MHz.
    """
    cr_times, splitted_data, dt = _get_normalized_cr_tomography_data(job_id,
                                                                     parallel_exp=parallel_exp,
                                                                     qubit_pair=qubit_pair)
    signal_x = splitted_data[:2]
    signal_y = splitted_data[2:4]
    signal_z = splitted_data[4:6]

    period0 = _estimate_period(signal_z[0], cr_times)
    period1 = _estimate_period(signal_z[1], cr_times)

    cutoff = -1
    _i = 0
    while True:
        try:
            ground_params1, ground_cov1 = recursive_fit(
                cr_times[:cutoff] * dt * 1.0e6,
                signal_x[0][:cutoff],
                signal_y[0][:cutoff],
                signal_z[0][:cutoff],
                p0=np.array([1, 1, 1, 1 / (period0 * dt * 1.0e6) * 2 * pi, 0.0]),
            )
            break
        except RuntimeError as e:
            _i += 1
            period0 *= 2
            if _i > 16:
                raise e
    ground_params2, ground_cov2, excited_params2, excited_cov2 = simple_fit(
        cr_times,
        signal_x,
        signal_y,
        signal_z,
    )
            

    excited_params1, excited_cov1 = recursive_fit(
        cr_times[:cutoff] * dt * 1.0e6,
        signal_x[1][:cutoff],
        signal_y[1][:cutoff],
        signal_z[1][:cutoff],
        p0=np.array([1, 1, 1, 1 / (period1 * dt * 1.0e6) * 2 * pi, 0.0]),
    )
    # ground_params, ground_cov = excited_params, excited_cov
    error1 = np.sum((signal_x[0] - _avg_X(cr_times*dt*1e6, *ground_params1)) **2) + \
             np.sum((signal_y[0] - _avg_Y(cr_times*dt*1e6, *ground_params1)) **2) + \
             np.sum((signal_z[0] - _avg_Z(cr_times*dt*1e6, *ground_params1)) **2) + \
             np.sum((signal_x[1] - _avg_X(cr_times*dt*1e6, *excited_params1))**2) + \
             np.sum((signal_y[1] - _avg_Y(cr_times*dt*1e6, *excited_params1))**2) + \
             np.sum((signal_z[1] - _avg_Z(cr_times*dt*1e6, *excited_params1))**2)
    error2 = np.sum((signal_x[0] - avg_X(cr_times, *ground_params2))** 2) + \
             np.sum((signal_y[0] - avg_Y(cr_times, *ground_params2))** 2) + \
             np.sum((signal_z[0] - avg_Z(cr_times, *ground_params2))** 2) + \
             np.sum((signal_x[1] - avg_X(cr_times, *excited_params2))**2) + \
             np.sum((signal_y[1] - avg_Y(cr_times, *excited_params2))**2) + \
             np.sum((signal_z[1] - avg_Z(cr_times, *excited_params2))**2)
    if error1 < error2:
        if show_plot:
            plot_cr_ham_tomo(
                cr_times * dt * 1.0e6,
                splitted_data,
                ground_params=ground_params1,
                excited_params=excited_params1,
                ground_cov=ground_cov1,
                excited_cov=excited_cov1,
            )
            plt.show()
        [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz(
            ground_params1, excited_params1, ground_cov1, excited_cov1
        )[0]
    else:
        if show_plot:
            plot_cr_ham_tomo_simple(
                cr_times,
                splitted_data,
                ground_params=ground_params2,
                excited_params=excited_params2,
                ground_cov=ground_cov2,
                excited_cov=excited_cov2,
                dt=dt,
            )
            plt.show()
        [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz_simple(
            ground_params2, excited_params2, ground_cov2, excited_cov2, dt=dt
        )[0]

    # if show_plot:
    #     plot_cr_ham_tomo_simple(
    #         cr_times,
    #         splitted_data,
    #         ground_params,
    #         excited_params,
    #         ground_cov,
    #         excited_cov,
    #         dt=dt,
    #     )
    #     plt.show()
    # [[IX, IY, IZ], [ZX, ZY, ZZ]] = get_interation_rates_MHz_simple(
    #     ground_pa rams, excited_params, ground_cov, excited_cov, dt=dt
    # )[0]
    return {"IX": IX, "IY": IY, "IZ": IZ, "ZX": ZX, "ZY": ZY, "ZZ": ZZ}

def _get_normalized_parallel_cr_tomography_data(job_id, qubit_pair=None):
  

    total_job_len = 0
    data = load_job_data(job_id[0])
    result = data["result"]
    qubit_pairs = data["parameters"]["qubit_pairs"]
    for job_i in job_id:
        data = load_job_data(job_i)
        result = data["result"]
        total_job_len += len(result.results)
    target_data_2d = np.zeros((len(qubit_pairs), total_job_len))

    current_job_num = 0
    num_qubit_pairs = len(qubit_pairs)
    dt = data["parameters"]["dt"]
    shots = data["parameters"]["shots"]
    cr_times = data["parameters"]["cr_times"]
    target_ids = [2 * num_qubit_pairs - 2 * i - 2 
        for i in range(num_qubit_pairs)]
    
    current_job_num = 0  # Initialize the job counter
   
    for job_i in job_id:
        data = load_job_data(job_i)
        result = data["result"]
    
    # Loop once over the result counts to minimize the nested structure
        for n in range(len(result.results)):
            counts = result.get_counts(n)  # Store counts once per n
        
        # Convert target ids to a mask and pre-compute the adjustment
            for state, count in counts.items():
                state_array = np.array([state[t_id] == "0" for t_id in target_ids], dtype=int)
                target_data_2d[:, current_job_num] += (state_array * count / shots)
        
            current_job_num += 1

    assert current_job_num == total_job_len
    for i in range(len(qubit_pairs)):
        target_data = target_data_2d[i]
        # print(target_data)
        target_data = 2 * target_data - 1
        scale = np.max(target_data) - np.min(target_data)
        average = (np.max(target_data) + np.min(target_data)) / 2
        target_data = 2 * (target_data - average) / scale
        target_data_2d[i] = target_data
        # _get_normalized_cr_tomography_data(job_id)
        # print(target_data_2d[i].reshape(len(cr_times), 6).transpose())
        # exit()
    
    # print(qubit_pairs) 
    if qubit_pair is None: 
        return cr_times, target_data_2d, dt
    else:
        try:
            qubit_pair_index = qubit_pairs.index(qubit_pair)
        except:
            qubit_pair_index = qubit_pairs.index(tuple(qubit_pair))
        return cr_times, target_data_2d[qubit_pair_index], dt


def get_normalized_parallel_cr_tomography_data(job_id, qubit_pair=None):
  

    total_job_len = 0
    data = load_job_data(job_id[0])
    result = data["result"]
    qubit_pairs = data["parameters"]["qubit_pairs"]
    for job_i in job_id:
        data = load_job_data(job_i)
        result = data["result"]
        total_job_len += len(result.results)
    target_data_2d = np.zeros((len(qubit_pairs), total_job_len))

    current_job_num = 0
    num_qubit_pairs = len(qubit_pairs)
    dt = data["parameters"]["dt"]
    shots = data["parameters"]["shots"]
    cr_times = data["parameters"]["cr_times"]
    target_ids = [2 * num_qubit_pairs - 2 * i - 2 
        for i in range(num_qubit_pairs)]
    
    current_job_num = 0  # Initialize the job counter
   
    for job_i in job_id:
        data = load_job_data(job_i)
        result = data["result"]
    
    # Loop once over the result counts to minimize the nested structure
        for n in range(len(result.results)):
            counts = result.get_counts(n)  # Store counts once per n
        
        # Convert target ids to a mask and pre-compute the adjustment
            for state, count in counts.items():
                state_array = np.array([state[t_id] == "0" for t_id in target_ids], dtype=int)
                target_data_2d[:, current_job_num] += (state_array * count / shots)
        
            current_job_num += 1

    assert current_job_num == total_job_len
    for i in range(len(qubit_pairs)):
        target_data = target_data_2d[i]
        # print(target_data)
        target_data = 2 * target_data - 1
        scale = np.max(target_data) - np.min(target_data)
        average = (np.max(target_data) + np.min(target_data)) / 2
        target_data = 2 * (target_data - average) / scale
        target_data_2d[i] = target_data
        # _get_normalized_cr_tomography_data(job_id)
        # print(target_data_2d[i].reshape(len(cr_times), 6).transpose())
        # exit()
    
    # print(qubit_pairs) 
    if qubit_pair is None: 
        return cr_times, target_data_2d, dt
    else:
        try:
            qubit_pair_index = qubit_pairs.index(qubit_pair)
        except:
            qubit_pair_index = qubit_pairs.index(tuple(qubit_pair))
        return cr_times, target_data_2d[qubit_pair_index], dt

    
    
    


def process_parallel_tomo_data(job_id, show_plot=False, qubit_pair=None):
    cr_times, target_data_2d, dt =  \
        _get_normalized_parallel_cr_tomography_data(job_id, qubit_pair)
    outdict_lst = []
    if qubit_pair is not None:
        target_data_2d = target_data_2d.reshape(-1, 6).transpose()
        out_dict = get_interact(cr_times, 
            target_data_2d, dt, show_plot=show_plot)
        return out_dict

    for _ in range(target_data_2d.shape[0]):
        splitted_data = target_data_2d[_].reshape(len(cr_times), 6).transpose()
        curren_dict = get_interact(cr_times, 
            splitted_data, dt, show_plot=show_plot)
        outdict_lst.append(curren_dict)
    return outdict_lst

def process_parallel_tomo_data_single(job_id, show_plot=False):
    cr_times, target_data_2d, dt = _get_normalized_parallel_cr_tomography_data(job_id)
    outdict_lst = []
    for _ in range(target_data_2d.shape[0]):
        splitted_data = target_data_2d[_].reshape(len(cr_times), 3).transpose()
        current_dict = get_interact_single(cr_times, splitted_data, dt, show_plot=show_plot)
        outdict_lst.append(current_dict)
    return outdict_lst
    
    

    
def plot_cr_ham_tomo_simple_single(
        cr_times,
        tomography_data,
        ground_params,
        ground_cov,
        dt
):
    colors = ["tab:blue", "tab:red"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 4), sharey=True)
    ax1.scatter(
        cr_times,
        tomography_data[0, :],
        lw=0.3,
        color=colors[0],
        label=None,
    )
    ax1.plot(cr_times, avg_X(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax2.scatter(
        cr_times,
        tomography_data[1, :],
        lw=0.3,
        color=colors[0],
        label=None,
    )
    ax2.plot(cr_times, avg_Y(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax3.scatter(
        cr_times,
        tomography_data[2, :],
        lw=0.3,
        color=colors[0],
        label=None,
    )
    ax3.plot(cr_times, avg_Z(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax3.set_xlabel(r"Time (dt)", fontsize="small")
    ax3.set_ylabel(r"$\langle Z(t) \rangle$", fontsize="small")
    ax3.set_yticklabels([])
    coeffs = ground_params / dt / 1.0e6 / 2 / pi
    errors = np.sqrt(np.diag(ground_cov)) / dt / 1.0e6 / 2 / pi
    ax3.text(
        cr_times[-1] / 2,
        -2.55,
        "IX = %.3f (%2.f) MHz   IY = %.3f (%2.f) MHz   IZ = %.3f (%2.f) MHz"
        % (
            coeffs[1],
            errors[1] * 1000,
            coeffs[2],
            errors[2] * 1000,
            coeffs[0],
            errors[0] * 1000,
        ),
        fontsize="x-small",
        horizontalalignment="center",
    )



def plot_cr_ham_tomo_simple(
    cr_times,
    tomography_data,
    ground_params,
    excited_params,
    ground_cov,
    excited_cov,
    dt
):
    colors = ["tab:blue", "tab:red"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 4), sharey=True)
    # is_single_qubit_tomo = excited_params is None
    

    ax1.scatter(
        cr_times,
        tomography_data[0, :],
        lw=0.3,
        color=colors[0],
        label="control in |0>",
    )
    ax1.plot(cr_times, avg_X(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax1.set_ylabel(r"$\langle X(t) \rangle$", fontsize="small")
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    ax1.scatter(
        cr_times,
        tomography_data[1, :],
        lw=0.3,
        color=colors[1],
        label="control in |1>",
    )
    ax1.plot(cr_times, avg_X(cr_times, *excited_params), lw=2.0, color=colors[1])
    ax1.legend(loc=4, fontsize="x-small")

    ax2.scatter(
        cr_times,
        tomography_data[2, :],
        lw=0.3,
        color=colors[0],
        label="control in |0>",
    )
    ax2.plot(cr_times, avg_Y(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax2.set_ylabel(r"$\langle Y(t) \rangle$", fontsize="small")
    ax2.set_xticklabels([])
    ax2.scatter(
        cr_times,
        tomography_data[3, :],
        lw=0.3,
        color=colors[1],
        label="control in |1>",
    )
    ax2.plot(cr_times, avg_Y(cr_times, *excited_params), lw=2.0, color=colors[1])
    ax2.legend(loc=4, fontsize="x-small")

    ax3.scatter(
        cr_times,
        tomography_data[4, :],
        lw=0.3,
        color=colors[0],
        label="control in |0>",
    )
    ax3.plot(cr_times, avg_Z(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax3.set_ylabel(r"$\langle Z(t) \rangle$", fontsize="small")
    ax3.set_yticklabels([])
    ax3.set_xlabel(r"Time (dt)", fontsize="small")
    ax3.scatter(
        cr_times,
        tomography_data[5, :],
        lw=0.3,
        color=colors[1],
        label="control in |1>",
    )
    ax3.plot(cr_times, avg_Z(cr_times, *excited_params), lw=2.0, color=colors[1])
    ax3.legend(loc=4, fontsize="x-small")

    coeffs, errors = get_interation_rates_MHz_simple(
        ground_params, excited_params, ground_cov, excited_cov, dt
    )

    ax3.text(
        cr_times[-1] / 2,
        -2.55,
        "ZX = %.3f (%2.f) MHz   ZY = %.3f (%2.f) MHz   ZZ = %.3f (%2.f) MHz"
        % (
            coeffs[1][0],
            errors[1][0] * 1000,
            coeffs[1][1],
            errors[1][1] * 1000,
            coeffs[1][2],
            errors[1][2] * 1000,
        ),
        fontsize="x-small",
        horizontalalignment="center",
    )

    ax3.text(
        cr_times[-1] / 2,
        -2.9,
        "IX = %.3f (%2.f) MHz   IY = %.3f (%2.f) MHz   IZ = %.3f (%2.f) MHz"
        % (
            coeffs[0][0],
            errors[0][0] * 1000,
            coeffs[0][1],
            errors[0][1] * 1000,
            coeffs[0][2],
            errors[0][2] * 1000,
        ),
        fontsize="x-small",
        horizontalalignment="center",
    )





def plot_cr_ham_tomo(
    cr_times,
    tomography_data,
    ground_params,
    excited_params=None,
    ground_cov=None,
    excited_cov=None,
):
    """Plot Hamiltonian tomography data and curve fits with interaction rates.

    Args:
        cr_times (np.ndarray): Array of CR times.
        tomography_data (np.ndarray): Averaged tomography data.
        ground_params (np.ndarray): Parameters of the ground fit.
        excited_params (np.ndarray, optional): Parameters of the excited fit. Default is None.
        ground_cov (np.ndarray, optional): Covariance matrix of the ground fit. Default is None.
        excited_cov (np.ndarray, optional): Covariance matrix of the excited fit. Default is None.
    """
    colors = ["tab:blue", "tab:red"]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(4, 4), sharey=True)
    is_single_qubit_tomo = excited_params is None

    if not is_single_qubit_tomo:
        ground_avg = tomography_data[0::2, :]
        excited_avg = tomography_data[1::2, :]
    else:
        ground_avg = tomography_data
    # Scatter plot and curve for X(t)
    ax1.scatter(
        cr_times,
        ground_avg[0, :],
        lw=0.3,
        color=colors[0],
        label=r"control in $|0\rangle$" if not is_single_qubit_tomo else None,
    )
    ax1.plot(cr_times, _avg_X(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax1.set_ylabel(r"$\langle X(t) \rangle$", fontsize="small")
    ax1.set_xticklabels([])
    ax1.set_yticks([])
    # Scatter plot and curve for Y(t)
    ax2.scatter(
        cr_times, ground_avg[1, :], lw=0.3, color=colors[0], label="control in |0>"
    )
    ax2.plot(cr_times, _avg_Y(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax2.set_ylabel(r"$\langle Y(t) \rangle$", fontsize="small")
    ax2.set_xticklabels([])
    # Scatter plot and curve for Z(t)
    ax3.scatter(
        cr_times,
        ground_avg[2, :],
        lw=0.3,
        color=colors[0],
        label=r"control in $|0\rangle$",
    )
    ax3.plot(cr_times, _avg_Z(cr_times, *ground_params), lw=2.0, color=colors[0])
    ax3.set_ylabel(r"$\langle Z(t) \rangle$", fontsize="small")
    ax3.set_yticklabels([])
    ax3.set_xlabel(r"Time (dt)", fontsize="small")

    if not is_single_qubit_tomo:
        ax1.scatter(
            cr_times,
            excited_avg[0, :],
            lw=0.3,
            color=colors[1],
            label=r"control in $|1\rangle$",
        )
        ax1.plot(cr_times, _avg_X(cr_times, *excited_params), lw=2.0, color=colors[1])
        ax1.legend(loc=4, fontsize="x-small")
        ax2.scatter(
            cr_times,
            excited_avg[1, :],
            lw=0.3,
            color=colors[1],
            label=r"control in $|1\rangle$",
        )
        ax2.plot(cr_times, _avg_Y(cr_times, *excited_params), lw=2.0, color=colors[1])
        ax3.scatter(
            cr_times,
            excited_avg[2, :],
            lw=0.3,
            color=colors[1],
            label=r"control in $|1\rangle$",
        )
        ax3.plot(cr_times, _avg_Z(cr_times, *excited_params), lw=2.0, color=colors[1])

    if not is_single_qubit_tomo:
        coeffs, errors = get_interation_rates_MHz(
            ground_params, excited_params, ground_cov, excited_cov
        )
        ax3.text(
            cr_times[-1] / 2,
            -2.55,
            "ZX = %.3f (%2.f) MHz   ZY = %.3f (%2.f) MHz   ZZ = %.3f (%2.f) MHz"
            % (
                coeffs[1][0],
                errors[1][0] * 1000,
                coeffs[1][1],
                errors[1][1] * 1000,
                coeffs[1][2],
                errors[1][2] * 1000,
            ),
            fontsize="x-small",
            horizontalalignment="center",
        )
        ax3.text(
            cr_times[-1] / 2,
            -2.9,
            "IX = %.3f (%2.f) MHz   IY = %.3f (%2.f) MHz   IZ = %.3f (%2.f) MHz"
            % (
                coeffs[0][0],
                errors[0][0] * 1000,
                coeffs[0][1],
                errors[0][1] * 1000,
                coeffs[0][2],
                errors[0][2] * 1000,
            ),
            fontsize="x-small",
            horizontalalignment="center",
        )
    else:
        coeffs = ground_params / 2 / pi
        errors = np.diagonal(ground_cov) ** 0.5 / 2 / pi
        ax3.text(
            cr_times[-1] / 2,
            -2.55,
            "X = %.3f (%2.f) MHz   Y = %.3f (%2.f) MHz   Z = %.3f (%2.f) MHz"
            % (
                coeffs[1],
                errors[1] * 1000,
                coeffs[2],
                errors[2] * 1000,
                coeffs[0],
                errors[0] * 1000,
            ),
            fontsize="x-small",
            horizontalalignment="center",
        )


# %% Iterative calibration
def _compute_drive_scale(
    coeff_dict_1, coeff_dict_2, ix_params, prob_ix_strength, conjugate_pulse
):
    vau1 = coeff_dict_1["IX"] + 1.0j * coeff_dict_1["IY"]
    vau2 = coeff_dict_2["IX"] + 1.0j * coeff_dict_2["IY"]
    A1 = ix_params["amp"] * np.exp(1.0j * ix_params["angle"])
    A2 = (ix_params["amp"] + prob_ix_strength) * np.exp(1.0j * ix_params["angle"])
    if conjugate_pulse:
        return (vau2 - vau1) / np.conjugate(A2 - A1)
    else:
        return (vau2 - vau1) / (A2 - A1)


def _angle(c):
    """
    User arctan to calculate the angle such that -1 won't be transfered to the angle parameters, in this way, a negative ZX will remains the same, not transfered to a positive ZX.

    The output range is (-pi/2, pi/2)
    """
    if c.real == 0.0:
        return np.pi / 2
    return np.arctan(c.imag / c.real)


def update_pulse_params(
    coeff_dict_1,
    coeff_dict_2,
    cr_params,
    ix_params,
    prob_ix_strength,
    target_IX_strength=None,
    backend_name=None,
    real_only=False,
):
    """
    Update pulse parameters for CR and IX gates based on tomography results. Refer to arxiv 2303.01427 for the derivation.

    Args:
        coeff_dict_1 (dict): Coefficients from the first tomography job.
        coeff_dict_2 (dict): Coefficients from the second tomography job.
        cr_params (dict): Parameters for the CR gate pulse.
        ix_params (dict): Parameters for the IX gate pulse.
        prob_ix_strength (float): The strength of the IX gate pulse.
        target_IX_strength (float, optional): Target angle for IX gate. Default is 0.
        backend_name (str):
        real_only: Update only the real part of the drive.

    Returns:
        tuple: Updated CR parameters, Updated IX parameters, None (for compatibility with CR-only calibration).
    """
    cr_params = cr_params.copy()
    ix_params = ix_params.copy()
    if backend_name == "DynamicsBackend":
        sign = -1  # dynamics end has a different convension
    else:
        sign = 1
    if "ZX" in coeff_dict_1:
        phi0 = _angle(coeff_dict_1["ZX"] + 1.0j * coeff_dict_1["ZY"])
    else:
        # No CR tomography, only single target qubit tomography.
        phi0 = 0.0
    cr_params["angle"] -= sign * phi0

    drive_scale = _compute_drive_scale(
        coeff_dict_1, coeff_dict_2, ix_params, prob_ix_strength, backend_name
    )
    logger.info(
        "Estimated drive scale: \n"
        f"{drive_scale/1000}" + "\n"
        f"{drive_scale/np.exp(1.j*_angle(drive_scale))/1000}" + "\n"
        f"{_angle(drive_scale)}"
    )

    # Update the IX drive strength and phase
    vau1 = coeff_dict_1["IX"] + 1.0j * coeff_dict_1["IY"]
    A1 = ix_params["amp"] * np.exp(sign * 1.0j * ix_params["angle"])
    new_drive = (target_IX_strength - vau1) / drive_scale + A1
    if real_only:
        ix_params["amp"] = np.real(new_drive)
    else:
        new_angle = _angle(new_drive)
        ix_params["amp"] = np.real(new_drive / np.exp(1.0j * new_angle))
        ix_params["angle"] = sign * (new_angle - phi0)
    if "ZX" in coeff_dict_1:
        return (
            cr_params,
            ix_params,
            None,
            np.real(drive_scale / np.exp(1.0j * _angle(drive_scale)) / 1000),
        )
    else:
        return (
            cr_params,
            None,
            ix_params,
            np.real(drive_scale / np.exp(1.0j * _angle(drive_scale)) / 1000),
        )


def _update_frequency_offset(old_calibration_data, mode, backend_name):
    """
    This should not be used separatly because applying more than once will lead to the wrong offset.
    """
    new_calibration_data = deepcopy(old_calibration_data)
    if mode == "CR":
        coeffs_dict = old_calibration_data["coeffs"]
        frequency_offset_key = "frequency_offset"
    else:
        coeffs_dict = old_calibration_data["x_gate_coeffs"]
        frequency_offset_key = "x_gate_frequency_offset"

    if backend_name == "DynamicsBackend":
        correction = coeffs_dict["IZ"] * 1.0e6
    else:
        correction = -0.5 * coeffs_dict["IZ"] * 1.0e6
    new_calibration_data[frequency_offset_key] = (
        old_calibration_data.get(frequency_offset_key, 0.0) + correction
    )
    if np.abs(coeffs_dict["IZ"]) > 2.0:
        logger.warning(
            "Frequency offset larger than 2MHz, update is not applied. Please check if the fit is accurate."
        )
        return old_calibration_data
    logger.info(
        f"Frequency offset is updated to {new_calibration_data[frequency_offset_key]} Hz"
    )
    return new_calibration_data


def iterative_cr_pulse_calibration(
    qubits,
    backend,
    cr_times,
    session,
    gate_name,
    initial_calibration_data=None,
    verbose=False,
    threshold_MHz=0.015,
    restart=False,
    rerun_last_calibration=True,
    max_repeat=4,
    shots=None,
    mode="CR",
    IX_ZX_ratio=None,
    save_result=True,
    control_states=None,
):
    """
    Iteratively calibrates CR pulses on the given qubits to remove the \
IX, ZY, IY terms. The result is saved in the carlibtaion data file \
and can be accessed via `read_calibration_data`.

    Args:
        qubits (list): Qubits involved in the calibration.
        backend: Quantum backend.
        cr_times (list): CR gate times for tomography.
        session: Qiskit runtime session.
        gate_name (str): Name of the CR gate. This will be used to identify the\
 calibration data when retrieving the information.
        initial_calibration_data (dict, optional): Initial parameters for\
 calibration.
        verbose (bool, optional): Whether to display detailed logging \
information. Default is False.
        threshold_MHz (float, optional): Error threshold for calibration. \
Default is 0.015 GHz.
        restart (bool, optional): Whether to restart calibration from \
scratch. Default is False.
        rerun_last_calibration (bool, optional): Whether to rerun the last \
calibration. Default is True.
        max_repeat (int, optional): Maximum number of calibration \
repetitions. Default is 4.
        shots (int, optional): Number of shots for each tomography job. \
Default is 1024.
        mode (str, optional): Calibration mode, "CR" or "IX-pi". \
Default is "CR". The CR mode is used to measure the ZX and ZZ strength. \
It only updates the phase of CR drive and the IX drive but not IY. \
The "IX-pi" mode updates the target drive for a CNOT gate.
    """
    if control_states is None:
        if mode == "CR":
            control_states = (0, 1)
        elif mode == "IX-pi":
            control_states = (1,)
        else:
            ValueError("Mode must be either 'CR' or 'IX-pi/2'.")
    if not restart:
        try:  # Load existing calibration
            logger.info("Loading existing calibration data...")
            qubit_calibration_data = read_calibration_data(backend, 
                                                            gate_name, 
                                                            (qubits))
            qubit_calibration_data = deepcopy(qubit_calibration_data)
            if mode == "IX-pi":
                qubit_calibration_data["x_gate_ix_params"]

        except KeyError:  #
            restart = True  # we need to overwrite rerun_last_calibration=True
            logger.warning(
                "Failed to find the calibration data for the " + \
                f"gate{backend.name, gate_name, (qubits)}. " + \
                "Restarting from scratch."
            )
    if restart:
        if not rerun_last_calibration:
            logger.warning(
                "Last calibration job for " + \
                f"{gate_name} not found or not used. Starting from scratch."
            )
            rerun_last_calibration = True
        if mode == "CR":
            if initial_calibration_data is None:
                raise ValueError(
                    "Starting calibration from scratch, " +\
                    "but initial parameters are not provided."
                )
            if (
                "cr_params" not in initial_calibration_data
                or "ix_params" not in initial_calibration_data
            ):
                raise ValueError(
                    "The initial pulse parameters for the CR "+
                    "and target drive must be provided."
                )
            qubit_calibration_data = initial_calibration_data
        else:
            logger.info("Loading existing calibration data...")
            qubit_calibration_data = read_calibration_data(backend, 
                                                           gate_name, 
                                                           (qubits))
            qubit_calibration_data = deepcopy(qubit_calibration_data)
            qubit_calibration_data["x_gate_ix_params"] = qubit_calibration_data[
                "ix_params"
            ].copy()
            qubit_calibration_data["x_gate_ix_params"]["amp"] = 0.0
            qubit_calibration_data["x_gate_ix_params"]["angle"] = 0.0
            if "beta" in qubit_calibration_data["x_gate_ix_params"]:
                del qubit_calibration_data["x_gate_ix_params"]["beta"]
            qubit_calibration_data["x_gate_frequency_offset"] = qubit_calibration_data[
                "frequency_offset"
            ]
    shots = 512 if shots is None else shots

    if mode == "CR" and IX_ZX_ratio is None:
        IX_ZX_ratio = 0.0
    elif mode == "IX-pi" and IX_ZX_ratio is None:
        IX_ZX_ratio = -2
    try:
        target_IX_strength = qubit_calibration_data["coeffs"]["ZX"] * IX_ZX_ratio
    except:
        target_IX_strength = 0.0
    logger.info(f"Target IX / ZX ratio: {IX_ZX_ratio}")
    logger.info(f"Target IX strength: {target_IX_strength} MHz")

    def _get_error(coeff_dict, mode, target_IX):
        if mode == "CR":
            error = np.array(
                (coeff_dict["IX"] - target_IX, coeff_dict["IY"], coeff_dict["ZY"])
            )
        elif mode == "IX-pi":
            error = np.array((coeff_dict["IX"] - target_IX, coeff_dict["IY"]))
        error = np.abs(error)
        max_error = np.max(error)
        return max_error

    def _error_smaller_than(coeff_dict, threshold_MHz, mode, target_IX=None):
        """
        Compare the measured coupling strength and check if the error terms are smaller than the threshold.

        Args:
            coeff_dict (dict): Coefficients from tomography job.
            threshold_MHz (float): Error threshold for calibration.
            mode (str): Calibration mode, "CR" or "IX-pi/2".
            target_IX (float, optional): Target IX angle. Default is None.

        Returns:
            bool: True if error is smaller than threshold, False otherwise.
        """
        if mode == "CR":
            error = np.array(
                (coeff_dict["IX"] - target_IX, coeff_dict["IY"], coeff_dict["ZY"])
            )
        elif mode == "IX-pi":
            error = np.array((coeff_dict["IX"] - target_IX, coeff_dict["IY"]))
        error = np.abs(error)
        max_error = np.max(error)
        error_type = ["IX", "IY", "ZY"][np.argmax(np.abs(error))]
        logger.info(f"Remaining dominant error: {error_type}: {max_error} MHz" + "\n")
        return max_error < threshold_MHz

    def _step_cr(qubit_calibration_data, n):
        """
        Submit two jobs, one with the given pulse parameter. 
        If the calibration is not finished, submit another one with 
        a shifted amplitude for the target drive. 
        This will be used to calculate a new set of 
        pulse parameters and returned.

        Args:
            qubit_calibration_data (dict): Calibration data.
            prob_ix_strength (float): Strength of the IX gate pulse.
            n (int): Calibration iteration number.

        Returns:
            tuple: Updated qubit_calibration_data, Calibration success flag.
        """
        cr_params = qubit_calibration_data["cr_params"]
        ix_params = qubit_calibration_data["ix_params"]
        x_gate_ix_params = None
        frequency_offset = qubit_calibration_data.get("frequency_offset", 0.0)
        if not rerun_last_calibration and n == 1:
            # If the last calibration was ran very recently, 
            # we can skip the first experiment that just 
            # rerun the tomography for the same pulse parameters.
            tomo_id1 = qubit_calibration_data["calibration_job_id"]
        else:
            # Run tomography experiment for the given parameters.
            tomo_id1 = send_cr_tomography_job(
                (qubits),
                backend,
                cr_params,
                ix_params,
                cr_times,
                x_gate_ix_params=x_gate_ix_params,
                frequency_offset=frequency_offset,
                blocking=True,
                session=session,
                shots=shots,
                control_states=control_states,
            )
        coeff_dict_1 = process_zx_tomo_data(tomo_id1, show_plot=verbose)

        target_IX_strength = (
            IX_ZX_ratio
            * np.sign(coeff_dict_1["ZX"])
            * np.sqrt(coeff_dict_1["ZX"] ** 2 + coeff_dict_1["ZY"] ** 2)
        )
        if verbose:
            logger.info("Tomography results for tomo_id1:\n" + 
                        str(coeff_dict_1) + "\n")

        qubit_calibration_data.update(
            {
                "calibration_job_id": tomo_id1,
                "coeffs": coeff_dict_1,
            }
        )

        # Interrupt the process if the calibration is successful or maximal repeat number is reached.
        if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
            if not (not rerun_last_calibration and n == 1):
                # print("cr_pulse line 2456, updating frequency offset:", n)
                qubit_calibration_data = _update_frequency_offset(
                    qubit_calibration_data, mode, backend.name
                )
        if _error_smaller_than(coeff_dict_1, threshold_MHz, mode, target_IX_strength):
            logger.info("Successfully calibrated.")
            return qubit_calibration_data, True
        if n > max_repeat:
            logger.info(
                f"Maximum repeat number {max_repeat} reached, calibration terminates."
            )
            return qubit_calibration_data, True

        # If not completed, send another job with shifted IX drive amplitude.
        # Remark: There is a strange observation that the 
        # Omega_GHz_amp_ratio measured here is not the same as the one 
        # estimated from single-qubit gate duration. There is a
        Omega_GHz_amp_ratio = qubit_calibration_data.get(
            "_omega_amp_ratio", amp_to_omega_GHz(backend, qubits[1], 1)
        )
        logger.info(f"Omega[GHz]/amp: {Omega_GHz_amp_ratio}")
        Omega_GHz_amp_ratio = np.real(Omega_GHz_amp_ratio)
        prob_ix_strength_MHz = target_IX_strength - coeff_dict_1["IX"]
        if np.abs(prob_ix_strength_MHz) > 0.1:  # Minimum 0.1 MHz
            prob_ix_strength = prob_ix_strength_MHz * 1.0e-3 / Omega_GHz_amp_ratio
            logger.info(f"Probe amp shift [MHz]: {prob_ix_strength_MHz} MHz")
        else:
            prob_ix_strength = (
                np.sign(prob_ix_strength_MHz) * 0.1e-3 / Omega_GHz_amp_ratio
            )
            logger.info("Probe amp shift [MHz]: 0.1 MHz")
        logger.info(f"Probe amp shift (amp): {prob_ix_strength}")

        tomo_id2 = shifted_parameter_cr_job(
            qubits,
            backend,
            cr_params,
            ix_params,
            cr_times,
            prob_ix_strength,
            x_gate_ix_params=x_gate_ix_params,
            frequency_offset=frequency_offset,
            blocking=True,
            session=session,
            shots=shots,
            control_states=control_states,
        )
        # print("after tomo_id2:", ix_params)
        coeff_dict_2 = process_zx_tomo_data(tomo_id2, show_plot=verbose)
        if verbose:
            logger.info(coeff_dict_2)
        # Compute the new parameters.
        (
            cr_params,
            updated_ix_params,
            updated_x_gate_ix_params,
            omega_amp_ratio,
        ) = update_pulse_params(
            coeff_dict_1,
            coeff_dict_2,
            cr_params,
            ix_params,
            prob_ix_strength,
            target_IX_strength=target_IX_strength,
            backend_name=backend.name,
            # only update the real part, imaginary part can be
            # unstable for small pulse ampliutude.
            real_only=True,
        )
        # This should not be added before the second experiment because it should only have a different IX drive amplitude.
        qubit_calibration_data.update(
            {
                "cr_params": cr_params,
                "ix_params": updated_ix_params,
                "_omega_amp_ratio": np.real(omega_amp_ratio),
            }
        )
        return qubit_calibration_data, False

    def _step_ix(qubit_calibration_data, n):
        """
        Submit two jobs, one with the given pulse parameter. If the calibration is not finished, submit another one with a shifted amplitude for the target drive. This will be used to calculate a new set of pulse parameters and returned.

        Args:
            qubit_calibration_data (dict): Calibration data.
            prob_ix_strength (float): Strength of the IX gate pulse.
            n (int): Calibration iteration number.

        Returns:
            tuple: Updated qubit_calibration_data, Calibration success flag.
        """
        if len(control_states) == 2:
            process_data_fun = process_zx_tomo_data
        else:
            process_data_fun = process_single_qubit_tomo_data
        cr_params = qubit_calibration_data["cr_params"]
        ix_params = qubit_calibration_data["ix_params"]
        x_gate_ix_params = qubit_calibration_data["x_gate_ix_params"]
        frequency_offset = qubit_calibration_data.get("x_gate_frequency_offset", 
                                                      0.0)
        if not rerun_last_calibration and n == 1:
            # If the last calibration was ran very recently, we can skip the first experiment that just rerun the tomography for the same pulse parameters.
            tomo_id1 = qubit_calibration_data["x_gate_calibration_job_id"]
        else:
            # Run tomography experiment for the given parameters.
            tomo_id1 = send_cr_tomography_job(
                (qubits),
                backend,
                cr_params,
                ix_params,
                cr_times,
                x_gate_ix_params=x_gate_ix_params,
                frequency_offset=frequency_offset,
                blocking=True,
                session=session,
                shots=shots,
                control_states=control_states,
            )
        coeff_dict_1 = process_data_fun(tomo_id1, show_plot=verbose)

        if verbose:
            logger.info("Tomography results:\n" + str(coeff_dict_1) + "\n")

        qubit_calibration_data.update(
            {
                "x_gate_calibration_job_id": tomo_id1,
                "x_gate_coeffs": coeff_dict_1,
            }
        )
        # Interrupt the process if the calibration is successful or maximal repeat number is reached.
        if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
            if not (not rerun_last_calibration and n == 1):
                qubit_calibration_data = _update_frequency_offset(
                    qubit_calibration_data, mode, backend.name
                )
        if _error_smaller_than(coeff_dict_1, threshold_MHz, mode, target_IX_strength):
            logger.info("Successfully calibrated.")
            return qubit_calibration_data, True
        if n > max_repeat:
            logger.info(
                f"Maximum repeat number {max_repeat} reached, calibration terminates."
            )
            return qubit_calibration_data, True

        # If not completed, send another job with shifted IX drive amplitude.
        Omega_GHz_amp_ratio = qubit_calibration_data.get(
            "_omega_amp_ratio", amp_to_omega_GHz(backend, qubits[1], 1)
        )
        Omega_GHz_amp_ratio = np.real(Omega_GHz_amp_ratio)
        logger.info(f"Omega[GHz]/amp: {Omega_GHz_amp_ratio}")
        prob_ix_strength_MHz = target_IX_strength - coeff_dict_1["IX"]
        if np.abs(prob_ix_strength_MHz) > 0.1:  # Minimum 0.1 MHz
            logger.info("Omega[GHz]/amp: " + str(Omega_GHz_amp_ratio))
            prob_ix_strength = prob_ix_strength_MHz * 1.0e-3 / Omega_GHz_amp_ratio
            logger.info(f"Probe amp shift [MHz]: {prob_ix_strength_MHz} MHz")
        else:
            logger.info("Omega[GHz]/amp: " + str(Omega_GHz_amp_ratio))
            prob_ix_strength = (
                np.sign(prob_ix_strength_MHz) * 0.1e-3 / Omega_GHz_amp_ratio
            )
            logger.info("Probe amp shift [MHz]: 0.1 MHz")
        logger.info(f"Probe amp shift (amp): {prob_ix_strength}")
        tomo_id2 = shifted_parameter_cr_job(
            qubits,
            backend,
            cr_params,
            ix_params,
            cr_times,
            prob_ix_strength,
            x_gate_ix_params=x_gate_ix_params,
            frequency_offset=frequency_offset,
            blocking=True,
            session=session,
            shots=shots,
            control_states=control_states,
            mode=mode,
        )
        coeff_dict_2 = process_data_fun(tomo_id2, show_plot=verbose)
        if verbose:
            logger.info(coeff_dict_2)
        # Compute the new parameters.
        cr_params, _, updated_x_gate_ix_params, omega_amp_ratio = update_pulse_params(
            coeff_dict_1,
            coeff_dict_2,
            cr_params,
            x_gate_ix_params,
            prob_ix_strength,
            target_IX_strength=target_IX_strength,
            backend_name=backend.name,
        )

        qubit_calibration_data.update(
            {
                "x_gate_ix_params": updated_x_gate_ix_params,
                "_omega_amp_ratio": np.real(omega_amp_ratio),
            }
        )
        return qubit_calibration_data, False

    succeed = False
    n = 1
    error = np.inf
    while (
        not succeed and n <= max_repeat + 1
    ):  # +1 because we need one last run for the calibration data.
        logger.info(f"\n\nCR calibration round {n}: ")
        if mode == "CR":
            qubit_calibration_data, succeed = _step_cr(qubit_calibration_data, n)
        else:
            qubit_calibration_data, succeed = _step_ix(qubit_calibration_data, n)
        target_IX_strength = qubit_calibration_data["coeffs"]["ZX"] * IX_ZX_ratio
        new_error = _get_error(
            qubit_calibration_data["coeffs"], mode, target_IX_strength
        )
        if save_result and new_error < error:
            save_calibration_data(
                backend, 
                gate_name, 
                qubits, 
                qubit_calibration_data)
            logger.info("CR calibration data saved.")
        n += 1
        shots = 2 * shots if shots < 2048 else shots
    if not succeed:
        logger.warnn(f"CR calibration failed after {n} round.")


def iy_drag_calibration(
    qubits,
    backend,
    gate_name,
    cr_times,
    session,
    verbose=False,
    threshold_MHz=0.015,
    delta_beta=None,
    shots=1024,
):
    """Calibrate the IY-DRAG pulse for the qubits and a precalibrated CR pulse.\
 It samples 3 "beta" value in the "ix_params" and perform an linear fit to \
obtain the correct IY-DRAG coefficient "beta" that zeros the ZZ interaction.

    Args:
        qubits (Tuple): Tuple containing the qubits involved in the gate.
        backend: The quantum backend.
        gate_name (str): Name of the gate for calibration. The pre calibrated \
CR pulse will be read from the database.
        cr_times (List): List of control pulse durations for CR experiments.
        session: Qiskit runtime session.
        verbose (bool, optional): Whether to display additional information. \
Defaults to False.
        threshold_MHz (float, optional): The error threshold for calibration in\
 MHz. Defaults to 0.015.
        delta_beta (float, optional): The step size for beta parameter \
calibration. Defaults to None.
    """
    logger.info("\n" + \
        f"Calibrating the IY-DRAG pulse for {qubits}-{gate_name}.")
    qubit_calibration_data = read_calibration_data(backend, gate_name, qubits)
    cr_params = qubit_calibration_data["cr_params"]
    ix_params = qubit_calibration_data["ix_params"]
    frequency_offset = qubit_calibration_data.get("frequency_offset", 0.0)

    # Sample three different IY strength.
    old_beta = ix_params.get("beta", 0.0)
    if "drag_type" in ix_params:
        ix_params["drag_type"] = "01"
        default_delta_beta = 2.0
    elif "drag_type" not in ix_params:
        default_delta_beta = 100.0
    else:
        raise ValueError("Unknown drag type.")
    delta_beta = (
        old_beta
        if (old_beta > default_delta_beta and delta_beta is None)
        else delta_beta
    )
    delta_beta = default_delta_beta if delta_beta is None else delta_beta
    beta_list = np.array([0.0, -delta_beta, delta_beta]) + old_beta

    ZZ_coeff_list = []
    for _, beta in enumerate(beta_list):
        if np.abs(beta - old_beta) < 1.0e-6:
            _shots = shots * 2
        else:
            _shots = shots
        ix_params["beta"] = beta
        job_id = send_cr_tomography_job(
            qubits,
            backend,
            cr_params,
            ix_params,
            cr_times,
            frequency_offset=frequency_offset,
            blocking=True,
            session=session,
            shots=_shots,
        )
        coeff_dict = process_zx_tomo_data(job_id, show_plot=verbose)
        ZZ_coeff_list.append(coeff_dict["ZZ"])
        if abs(beta - old_beta) < 1.0e-5 and abs(coeff_dict["ZZ"]) < threshold_MHz:
            logger.info(
                f"ZZ error {round(coeff_dict['ZZ'], 3)} MHz, no need for further calibration."
            )
            qubit_calibration_data.update(
                {
                    "calibration_job_id": job_id,
                    "coeffs": coeff_dict,
                }
            )
            save_calibration_data(backend, gate_name, qubits, qubit_calibration_data)
            if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
                qubit_calibration_data = _update_frequency_offset(
                    qubit_calibration_data, "CR", backend.name
                )
            return

    logger.info(f"ZZ sampling measurements complete : {ZZ_coeff_list}." + "\n")

    # Fit a linear curve.
    fun = lambda x, a, b: a * x + b
    par, _ = curve_fit(fun, beta_list, ZZ_coeff_list)
    calibrated_beta = -par[1] / par[0]
    logger.info(f"Calibrated IY beta: {calibrated_beta}" + "\n")

    if verbose:
        fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
        plt.scatter(beta_list, ZZ_coeff_list)
        x_line = np.linspace(min(beta_list), max(beta_list))
        y_line = fun(x_line, *par)
        plt.plot(x_line, y_line)
        plt.xlabel("beta")
        plt.ylabel("ZZ [MHz]")
        plt.show()

    # Perform a final tomography measurement.
    ix_params["beta"] = calibrated_beta
    job_id = send_cr_tomography_job(
        qubits,
        backend,
        cr_params,
        ix_params,
        cr_times,
        frequency_offset=frequency_offset,
        blocking=True,
        session=session,
        shots=shots * 2,
    )

    # Compute the interaction strength and save the calibration data.
    coeff_dict = process_zx_tomo_data(job_id, show_plot=verbose)
    logger.info(f"Updated coupling strength: {coeff_dict}")
    qubit_calibration_data.update(
        {
            "calibration_job_id": job_id,
            "coeffs": coeff_dict,
            "ix_params": ix_params,
        }
    )
    if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
        qubit_calibration_data = _update_frequency_offset(
            qubit_calibration_data, "CR", backend.name
        )

    save_calibration_data(backend, gate_name, qubits, qubit_calibration_data)
    logger.info(f"IY-DRAG calibration complete, new calibration data saved.")

def _generate_parallel_circuits(
        duration,
        qubit_pairs,
        backend,
        cr_params_lst,
        ix_params_lst,
        x_gate_ix_params_lst,
        frequency_offset_lst,
        control_states,
):
    cr_gate = circuit.Gate("cr", num_qubits=2, params=[])
    circ_list = []
    # for qubit_pair in qubit_pairs:
        # (qc, qt) = qubit_pair
    cr_sched_lst = []
    qc_lst = []
    qt_lst = []
    for i, qubit_pair in enumerate(qubit_pairs):
        (qc, qt) = qubit_pair
        qc_lst.append(qc)
        qt_lst.append(qt)
        cr_params = cr_params_lst[i]
        cr_params["duration"] = int(duration)
        ix_params = ix_params_lst[i]
        ix_params["duration"] = int(duration)
        x_gate_ix_params = x_gate_ix_params_lst[i]
        if x_gate_ix_params is not None:
            # print(x_gate_ix_params)
            x_gate_ix_params["duration"] = int(duration)
        
        frequency_offset = frequency_offset_lst[i]
        cr_sched = get_cr_schedule(
            (qc, qt),
            backend,
            cr_params = cr_params,
            ix_params = ix_params,
            x_gate_ix_params = x_gate_ix_params,
            frequency_offset = frequency_offset,
        )
        cr_sched_lst.append(cr_sched)
    for basis in ["X", "Y", "Z"]:
        for control in control_states:
            tomo_circ = circuit.QuantumCircuit(
                backend.num_qubits,
                2 * len(qubit_pairs),
            )
            if control in (1, "1"):
                tomo_circ.x(qc_lst)
            tomo_circ.append(cr_gate, [qc_lst, qt_lst])
            tomo_circ.barrier(qc_lst + qt_lst)
            if basis == "X":
                tomo_circ.rz(np.pi / 2, qt_lst)
                tomo_circ.sx(qt_lst)
            elif basis == "Y":
                tomo_circ.sx(qt_lst)
            tomo_circ.measure(qc_lst, [2 * i for i in range(len(qubit_pairs))])
            tomo_circ.measure(qt_lst, [2 * i + 1 for i in range(len(qubit_pairs))])
            for i, cr_sched in enumerate(cr_sched_lst):
                tomo_circ.add_calibration(gate=cr_gate, 
                                          qubits=(qc_lst[i], qt_lst[i]), 
                                          schedule=cr_sched)
            circ_list.append(tomo_circ)
    return circ_list
            






    
def parallel_get_cr_tomo_circuits(
    qubit_pairs: list[list[int]],
    backend,
    cr_times,
    cr_params_lst,
    ix_params_lst,
    x_gate_ix_params_lst,
    frequency_offset_lst,
    control_states=(0, 1),
):
    tomo_circs = []
    # print("cr_pulse line 2685:", x_gate_ix_params_lst)
    tmp_fun = partial(
        _generate_parallel_circuits,
        qubit_pairs=qubit_pairs,
        backend=backend,
        cr_params_lst=cr_params_lst,
        ix_params_lst=ix_params_lst,
        x_gate_ix_params_lst=x_gate_ix_params_lst,
        frequency_offset_lst=frequency_offset_lst,
        control_states=control_states,
    )

    tomo_circs = map(tmp_fun, cr_times)
    tomo_circs = sum(tomo_circs, [])
    logger.info(f"Generated {len(tomo_circs)} tomography circuits.")
    return tomo_circs











def parallel_send_cr_tomography_job(
    qubits: list[list[int]],
    backend,
    cr_params_lst,
    ix_params_lst,
    cr_times,
    frequency_offset_lst,
    x_gate_ix_params_lst=None,
    blocking=False,
    shots=1024,
    control_states=(0, 1),
    decoy = False,
):
    if x_gate_ix_params_lst is None:
        x_gate_ix_params_lst = [None] * len(qubits)
    cr_tomo_circ_list = parallel_get_cr_tomo_circuits(
        qubits,
        backend,
        cr_times,
        cr_params_lst,
        ix_params_lst,
        x_gate_ix_params_lst,
        frequency_offset_lst,
        control_states,
    )

    transpiled_circs = transpile(cr_tomo_circ_list,
                                    backend,
                                    optimization_level=1)
    # for circ in transpiled_circs:
    #     dag = circuit_to_dag(circ)
    #     idle_wires = list(dag.idle_wires())
    #     for wire in idle_wires:
    #         for qubit in qubits:
    #             if wire._index in qubit:
    #                 raise ValueError(
    #                     f"Qubit {wire._index} is idle but it is in the qubit list."
    #                 )
    # batch = Batch()
    # # batch = Batch(backend=backend)
    # sampler = Sampler(mode=batch)
    pub_circ = [(circ) for circ in transpiled_circs]
    max_circuits = 5
    all_partitioned_circuits = []
    for i in range(0, len(pub_circ), max_circuits):
        all_partitioned_circuits.append(pub_circ[i : i + max_circuits])
    jobs = []
    retry_num = 0
    max_repeat = 3
    while True:
        with Batch(backend=backend):
            sampler = Sampler()
            for partitioned_circuits in all_partitioned_circuits:
                job = sampler.run(partitioned_circuits)
                jobs.append(job)
        if sanity_check(jobs):
            break
        else:
            if retry_num >= max_repeat:
                send_tele_msg("CR tomography sanity check failed.")
                break
            retry_num += 1
            send_tele_msg("CR tomography sanity check failed. Retrying for " +
                          f"{retry_num} times.")
            
        



    # job = sampler.run(
    #     pub_circ,
    # )
    tag = "CR-tomo"
    parameters = {
        "backend": backend,
        "qubit_pairs": qubits,
        "cr_times": cr_times,
        "shots": shots,
        "cr_params": cr_params_lst,
        "ix_params": ix_params_lst,
        "x_gate_ix_params": x_gate_ix_params_lst,
        "frequency_offset": frequency_offset_lst,
        "dt": backend.dt,
    }
    logger.info(
        tag
        + ": "
        + ' '.join([job.job_id() for job in jobs])
        + "\n"
        + "\n".join([f"{key}: {val}" for key, val in parameters.items()])
        + "\n"
    )
    if blocking:
        save_parallel_job_data(jobs, backend=backend, parameters=parameters)
    else:
        async_execute(save_job_data, job, backend=backend, parameters=parameters)
    return [job.job_id() for job in jobs]

    









def parallel_cr_pulse_calib(
        qubits: list[list[int]],
        backend,
        cr_times,
        gate_name,
        initial_calibration_data=None,
        verbose=False,
        threshold_MHz=0.015,
        restart=False,
        rerun_last_calibration=True,
        max_repeat=4,
        shots=None,
        mode="CR",
        IX_ZX_ratio=None,
        save_result=True,
        control_states=None,
):
    '''
    Parallel calibration of CR pulses for multiple qubits.
    '''
    qubit_calibration_data_lst = []
    if control_states is None:
        if mode == "CR":
            control_states = (0, 1)
        elif mode == "IX-pi":
            control_states = (1,)
        else:
            ValueError("Mode must be either 'CR' or 'IX-pi/2'.")
    if not restart:
        try:  # Load existing calibration
            logger.info("Loading existing calibration data...")
            for qubit in qubits:
                qubit_calibration_data = read_calibration_data(backend, 
                                                                gate_name, 
                                                                qubit)
                qubit_calibration_data = deepcopy(qubit_calibration_data)
                if mode == "IX-pi":
                    qubit_calibration_data["x_gate_ix_params"]
                qubit_calibration_data_lst.append(qubit_calibration_data)
        except KeyError:  #
            restart = True
            logger.warning(
                "Failed to find the calibration data for the " + \
                f"gate{backend.name, gate_name, (qubits)}. " + \
                "Restarting from scratch."
            )
    if restart:
        if not rerun_last_calibration:
            logger.warning(
                "Last calibration job for " + \
                f"{gate_name} not found or not used. Starting from scratch."
            )
            rerun_last_calibration = True
        if mode == "CR":
            if initial_calibration_data is None:
                raise ValueError(
                    "Starting calibration from scratch, " +\
                    "but initial parameters are not provided."
                )
            for initial_calibration in initial_calibration_data:
                if (
                    "cr_params" not in initial_calibration
                    or "ix_params" not in initial_calibration
                ):
                    raise ValueError(
                        "The initial pulse parameters for the CR "+
                        "and target drive must be provided."
                    )
            for initial_calibration in initial_calibration_data:
                qubit_calibration_data = initial_calibration
                qubit_calibration_data_lst.append(qubit_calibration_data)
        else:
            logger.info("Loading existing calibration data...")
            for qubit in qubits:
                qubit_calibration_data = read_calibration_data(backend, 
                                                               gate_name, 
                                                               tuple(qubit)
                                                            )
                qubit_calibration_data = deepcopy(qubit_calibration_data)
                qubit_calibration_data["x_gate_ix_params"] = qubit_calibration_data[
                    "ix_params"
                ].copy()
                qubit_calibration_data["x_gate_ix_params"]["amp"] = 0.0
                qubit_calibration_data["x_gate_ix_params"]["angle"] = 0.0
                if "beta" in qubit_calibration_data["x_gate_ix_params"]:
                    del qubit_calibration_data["x_gate_ix_params"]["beta"]
                qubit_calibration_data["x_gate_frequency_offset"] = qubit_calibration_data[
                    "frequency_offset"
                ]
                qubit_calibration_data_lst.append(qubit_calibration_data)
    shots = 512 if shots is None else shots

    if mode == "CR" and IX_ZX_ratio is None:
        IX_ZX_ratio = 0.0
    elif mode == "IX-pi" and IX_ZX_ratio is None:
        IX_ZX_ratio = -2
    target_IX_strength_lst = np.zeros(len(qubits))
    try:
        for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
            target_IX_strength_lst[i] = qubit_calibration_data["coeffs"]["ZX"] * IX_ZX_ratio
    except:
        pass
    logger.info(f"Target IX / ZX ratio: {IX_ZX_ratio}")
    logger.info(f"Target IX strength: {target_IX_strength_lst}")

    def _get_error(coeff_dict, mode, target_IX):
        if mode == "CR":
            error = np.array(
                (coeff_dict["IX"] - target_IX, coeff_dict["IY"], coeff_dict["ZY"])
            )
        elif mode == "IX-pi":
            error = np.array((coeff_dict["IX"] - target_IX, coeff_dict["IY"]))
        error = np.abs(error)
        max_error = np.max(error)
        return max_error

    def _error_smaller_than(coeff_dict, threshold_MHz, mode, target_IX=None):
        """
        Compare the measured coupling strength and check if the error terms 
        are smaller than the threshold.

        Args:
            coeff_dict (dict): Coefficients from tomography job.
            threshold_MHz (float): Error threshold for calibration.
            mode (str): Calibration mode, "CR" or "IX-pi/2".
            target_IX (float, optional): Target IX angle. Default is None.

        Returns:
            bool: True if error is smaller than threshold, False otherwise.
        """
        if mode == "CR":
            error = np.array(
                (coeff_dict["IX"] - target_IX, 
                 coeff_dict["IY"], 
                 coeff_dict["ZY"])
            )
        elif mode == "IX-pi":
            error = np.array((coeff_dict["IX"] - target_IX, coeff_dict["IY"]))
        error = np.abs(error)
        max_error = np.max(error)
        error_type = ["IX", "IY", "ZY"][np.argmax(np.abs(error))]
        logger.info(f"Remaining dominant error: {error_type}: {max_error} MHz" + "\n")
        return max_error < threshold_MHz

    def _parallel_step_cr(qubit_calibration_data_lst, n, finished):
        cr_params_lst = []
        ix_params_lst = []
        x_gate_ix_params = None
        frequency_offset_lst = []
        for qubit_calibration_data in qubit_calibration_data_lst:
            cr_params = qubit_calibration_data["cr_params"]
            ix_params = qubit_calibration_data["ix_params"]
            frequency_offset = qubit_calibration_data.get("frequency_offset", 0.0)
            cr_params_lst.append(cr_params)
            ix_params_lst.append(ix_params)
            frequency_offset_lst.append(frequency_offset)
        if not rerun_last_calibration and n == 1:
            # If the last calibration was ran very recently, 
            # we can skip the first experiment that just 
            # rerun the tomography for the same pulse parameters.
            tomo_id1_lst = [qubit_calibration_data["calibration_job_id"] for qubit_calibration_data in qubit_calibration_data_lst]
        else:
            tomo_id = parallel_send_cr_tomography_job(
                qubits,
                backend,
                cr_params_lst,
                ix_params_lst,
                cr_times,
                frequency_offset_lst,
                blocking=True,
                shots=shots,
                control_states=control_states,
                x_gate_ix_params_lst=[x_gate_ix_params] * len(qubits),
            )
        coeff_lst = process_parallel_tomo_data(
            tomo_id, show_plot=verbose
        )
        if verbose:
            for coeff_dict in coeff_lst:
                logger.info("Tomography results for tomo_id1:\n" 
                            + str(coeff_dict) + "\n")
        omega_ghz_amp_ratio_lst = []
        prob_ix_strength_lst = []
        current_time = get_total_time(tomo_id)
        for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
            qubit_calibration_data.update(
                {
                    "calibration_job_id": tomo_id,
                    "coeffs": coeff_lst[i],
                }
            )
            if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
                # print("line 3199, start updating frequency offset")
                if not (not rerun_last_calibration and n == 1):
                    qubit_calibration_data = _update_frequency_offset(
                        qubit_calibration_data, mode, backend.name
                    )
            else:
                logger.info(f"no need to update frequency offset for qubit {qubits[i]} at round {n}")
            if _error_smaller_than(coeff_lst[i], threshold_MHz, mode, target_IX_strength_lst[i]):
                logger.info(f"Successfully calibrated Qubits: {qubits[i]}.")
                if qubits[i] not in finished:
                    finished.append(qubits[i])
                if len(finished) == len(qubits):
                    return qubit_calibration_data_lst, True, finished
                # continue
            if n > max_repeat:
                logger.info(
                    f"Maximum repeat number {max_repeat} reached, calibration terminates."
                )
                return qubit_calibration_data_lst, True, finished
            omega_ghz_amp_ratio = qubit_calibration_data.get(
                "_omega_amp_ratio", amp_to_omega_GHz(backend, qubits[i][1], 1)
            )
            omega_ghz_amp_ratio = np.real(omega_ghz_amp_ratio)
            omega_ghz_amp_ratio_lst.append(omega_ghz_amp_ratio)
            logger.info(f"Omega[GHz]/amp: {omega_ghz_amp_ratio}")
            prob_ix_strength_MHz = target_IX_strength_lst[i] - coeff_lst[i]["IX"]
            if qubits[i] in finished:
                prob_ix_strength = 0.0
                logger.info(f"Qubit {qubits[i]} already finished.")
            elif np.abs(prob_ix_strength_MHz) > 0.1:
                prob_ix_strength = prob_ix_strength_MHz * 1.0e-3 / omega_ghz_amp_ratio
                logger.info("Probe amp shift [MHz]" +
                            f"for Qubits {qubits[i]}:" +
                            f"{prob_ix_strength_MHz} MHz" + "\n" +
                            f"Probe amp shift (amp) for Qubits {qubits[i]}: {prob_ix_strength}"
                            )
            else:
                prob_ix_strength = (
                    np.sign(prob_ix_strength_MHz) * 0.1e-3 / omega_ghz_amp_ratio
                )
                logger.info("Probe amp shift [MHz]"+
                            f"for Qubits {qubits[i]}: 0.1 MHz" + "\n" +
                            f"Probe amp shift (amp) for Qubits {qubits[i]}: {prob_ix_strength}")
            if qubits[i] not in finished:
                if "calibration_time" in qubit_calibration_data:
                    qubit_calibration_data["calibration_time"] += current_time
                else:
                    qubit_calibration_data["calibration_time"] = current_time
            logger.info(f"Probe amp shift (amp) for Qubits {qubits[i]}: {prob_ix_strength}")
            prob_ix_strength_lst.append(prob_ix_strength)
            qubit_calibration_data_lst[i] = qubit_calibration_data
        
        logger.info(f"Probe amp shift length {len(prob_ix_strength_lst)}")
        logger.info(f"Probe amp shift {prob_ix_strength_lst}")

        # print("line 3250, before tomo_id2", qubit_calibration_data_lst)
        tomo_id2 = shifted_parallel_parameter_cr_job(
            qubits,
            backend,
            cr_params_lst,
            ix_params_lst,
            cr_times,
            prob_ix_strength_lst,
            x_gate_ix_params_lst=[x_gate_ix_params] * len(qubits),
            frequency_offset_lst=frequency_offset_lst,
            blocking=True,
            shots=shots,
            control_states=control_states,
            mode=mode,
        )
        # print("after tomo_id2:", ix_params_lst)
        coeff_lst_2 = process_parallel_tomo_data(tomo_id2, show_plot=False)
        print("finished tomo_id2")
        current_time2 = get_total_time(tomo_id2)
        if verbose:
            coeff_lst_2 = process_parallel_tomo_data(tomo_id2, show_plot=verbose)
            logger.info(coeff_lst_2)
        for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
            if qubits[i] in finished:
                continue
            qubit_calibration_data["calibration_time"] += current_time2
            # print("line 3267, start updating based on tomo_id2")
            (
                cr_params,
                updated_ix_params,
                updated_x_gate_ix_params,
                omega_amp_ratio,
            ) = update_pulse_params(
                coeff_lst[i],
                coeff_lst_2[i],
                cr_params_lst[i],
                ix_params_lst[i],
                prob_ix_strength_lst[i],
                target_IX_strength=target_IX_strength_lst[i],
                backend_name=backend.name,
                real_only=True,
            )
            qubit_calibration_data.update(
                {
                    "cr_params": cr_params,
                    "ix_params": updated_ix_params,
                    "_omega_amp_ratio": np.real(omega_amp_ratio),
                }
            )
            # print("line 3287", qubit_calibration_data["frequency_offset"])
        return qubit_calibration_data_lst, False, finished
    
    def _parallel_step_ix(qubit_calibration_data_lst, n, finished):
        if len(control_states) == 2:
            process_data_fun = process_parallel_tomo_data
        else:
            process_data_fun = process_parallel_tomo_data_single
        cr_params_lst = []
        ix_params_lst = []
        x_gate_ix_params_lst = []
        frequency_offset_lst = []
        for qubit_calibration_data in qubit_calibration_data_lst:
            cr_params = qubit_calibration_data["cr_params"]
            ix_params = qubit_calibration_data["ix_params"]
            x_gate_ix_params = qubit_calibration_data["x_gate_ix_params"]
            frequency_offset = qubit_calibration_data.get("x_gate_frequency_offset", 
                                                          0.0)
            cr_params_lst.append(cr_params)
            ix_params_lst.append(ix_params)
            x_gate_ix_params_lst.append(x_gate_ix_params)
            frequency_offset_lst.append(frequency_offset)
        
        if not rerun_last_calibration and n == 1:
            # If the last calibration was ran very recently, we can skip the first experiment that just rerun the tomography for the same pulse parameters.
            tomo_id1_lst = [qubit_calibration_data["x_gate_calibration_job_id"] 
                            for qubit_calibration_data in qubit_calibration_data_lst]
        else:
            tomo_id = parallel_send_cr_tomography_job(
                qubits,
                backend,
                cr_params_lst,
                ix_params_lst,
                cr_times,
                frequency_offset_lst,
                x_gate_ix_params_lst=x_gate_ix_params_lst,
                blocking=True,
                shots=shots,
                control_states=control_states,
            )
        # print("line 3449 ", tomo_id)
        coeff_lst_1 = process_data_fun(tomo_id, show_plot=verbose)
        if verbose:
            for coeff_dict in coeff_lst_1:
                logger.info("Tomography results:\n" + str(coeff_dict) + "\n")
        # finished = []
        for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
            qubit_calibration_data.update(
                {
                    "x_gate_calibration_job_id": tomo_id,
                    "x_gate_coeffs": coeff_lst_1[i],
                }
            )
            if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
                if not (not rerun_last_calibration and n == 1):
                    qubit_calibration_data = _update_frequency_offset(
                        qubit_calibration_data, mode, backend.name
                    )
            if _error_smaller_than(coeff_lst_1[i], threshold_MHz, mode, target_IX_strength_lst[i]):
                logger.info(f"Successfully calibrated Qubits: {qubits[i]}.")
                if qubits[i] not in finished:
                    finished.append(qubits[i])
                if len(finished) == len(qubits):
                    return qubit_calibration_data_lst, True, finished
            if n > max_repeat:
                logger.info(
                    f"Maximum repeat number {max_repeat} reached, calibration terminates."
                )
                return qubit_calibration_data_lst, True, finished
        
        Omega_GHz_amp_ratio_lst = []
        prob_ix_strength_lst = []
        current_time = get_total_time(tomo_id)
        for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
            Omega_GHz_amp_ratio = qubit_calibration_data.get(
                "_omega_amp_ratio", amp_to_omega_GHz(backend, qubits[i][1], 1)
            )
            Omega_GHz_amp_ratio = np.real(Omega_GHz_amp_ratio)
            Omega_GHz_amp_ratio_lst.append(Omega_GHz_amp_ratio)
            prob_ix_strength_MHz = target_IX_strength_lst[i] - coeff_lst_1[i]["IX"]
            if qubits[i] in finished:
                prob_ix_strength = 0.0
                logger.info(f"Qubit {qubits[i]} is finished.")
                prob_ix_strength_lst.append(prob_ix_strength)
            elif np.abs(prob_ix_strength_MHz) > 0.1:
                logger.info("Omega[GHz]/amp: " + f"{Omega_GHz_amp_ratio}")
                prob_ix_strength = prob_ix_strength_MHz * 1.0e-3 / Omega_GHz_amp_ratio
                logger.info("Probe amp shift [MHz]" +
                            f"for Qubits {qubits[i]}: " +
                            f"{prob_ix_strength_MHz} MHz" + "\n" +
                            f"Probe amp shift (amp) for Qubits {qubits[i]}: {prob_ix_strength}")
                prob_ix_strength_lst.append(prob_ix_strength)
            else:
                logger.info("Omega[GHz]/amp: " + f"{Omega_GHz_amp_ratio}")
                prob_ix_strength = (
                    np.sign(prob_ix_strength_MHz) * 0.1e-3 / Omega_GHz_amp_ratio
                )
                logger.info(f"Probe amp shift [MHz]" +
                            f"for Qubits {qubits[i]}: 0.1 MHz" + "\n" +
                            f"Probe amp shift (amp) for Qubits {qubits[i]}: {prob_ix_strength}")
                prob_ix_strength_lst.append(prob_ix_strength)
            if qubits[i] not in finished:
                if "calibration_time" in qubit_calibration_data:
                    qubit_calibration_data["calibration_time"] += current_time
                else:
                    qubit_calibration_data["calibration_time"] = current_time
        # print("prob_ix_strength_lst: line 3319", prob_ix_strength_lst)
        tomo_id2 = shifted_parallel_parameter_cr_job(
            qubits,
            backend,
            cr_params_lst,
            ix_params_lst,
            cr_times,
            prob_ix_strength_lst,
            x_gate_ix_params_lst=x_gate_ix_params_lst,
            frequency_offset_lst=frequency_offset_lst,
            blocking=True,
            shots=shots,
            control_states=control_states,
            mode=mode,
        )
        coeff_lst_2 = process_data_fun(tomo_id2, show_plot=verbose)
        if verbose:
            for coeff_dict in coeff_lst_2:
                logger.info(coeff_dict)
        current_time2 = get_total_time(tomo_id2)
        for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
            if qubits[i] in finished:
                continue
            qubit_calibration_data["calibration_time"] += current_time2
            (
                cr_params,
                _,
                updated_x_gate_ix_params,
                omega_amp_ratio,
            ) = update_pulse_params(
                coeff_lst_1[i],
                coeff_lst_2[i],
                cr_params_lst[i],
                x_gate_ix_params_lst[i],
                prob_ix_strength_lst[i],
                target_IX_strength=target_IX_strength_lst[i],
                backend_name=backend.name,
            )
            qubit_calibration_data.update(
                {
                    "x_gate_ix_params": updated_x_gate_ix_params,
                    "_omega_amp_ratio": np.real(omega_amp_ratio),
                }
            )
        return qubit_calibration_data_lst, False, finished
    
    succeed = False
    n = 1
    error_lst = np.inf * np.ones(len(qubits))
    finished = []
    while (
        not succeed and n <= max_repeat + 1
    ):  # +1 because we need one last run for the calibration data.
        logger.info(f"\n\nCR calibration round {n}: ")
        if mode == "CR":
            qubit_calibration_data_lst, succeed, curr_finished \
            = _parallel_step_cr(qubit_calibration_data_lst, n, finished)
            logger.info(f"Finished qubits: {curr_finished}")
            finished = [list(x) for x in set(tuple(sublist) for sublist in (finished + curr_finished))]
        else:
            qubit_calibration_data_lst, succeed, curr_finished \
            = _parallel_step_ix(qubit_calibration_data_lst, n, finished)
            logger.info(f"Finished qubits: {curr_finished}")
            finished = [list(x) for x in set(tuple(sublist) for sublist in (finished + curr_finished))]
        new_error_lst = np.array(
            [
                _get_error(qubit_calibration_data["coeffs"], mode, target_IX_strength_lst[i])
                for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst)
            ]
        )
        if save_result:
            for i, qubit_calibration_data in enumerate(qubit_calibration_data_lst):
                if new_error_lst[i] < error_lst[i]:
                    save_calibration_data(
                        backend, gate_name, tuple(qubits[i]), qubit_calibration_data
                    )
                logger.info("CR calibration data saved for qubits: " +
                            f"{qubits[i]}")
        n += 1
        shots = 2 * shots if shots < 2048 else shots
        # shots = 512
    if not succeed:
        logger.warnn(f"CR calibration failed after {n} round.")
    return finished


    

def parallel_iy_drag_calibration(
        qubit_pairs: list[list[int]],
        backend,
        gate_name,
        cr_times,
        verbose=False,
        threshold_MHz=0.015,
        delta_beta=None,
        shots=1024,
):
    cr_params_lst = []
    ix_params_lst = []
    frequency_offset_lst = []
    beta_list_lst = []
    old_beta_lst = []
    for qubit_pair in qubit_pairs:
        logger.info("\n" + \
            f"Calibrating the IY-DRAG pulse for {qubit_pair}-{gate_name}.")
        qubit_calibration_data = read_calibration_data(backend, 
                                                       gate_name, 
                                                       tuple(qubit_pair))
        cr_params = qubit_calibration_data["cr_params"]
        ix_params = qubit_calibration_data["ix_params"]
        frequency_offset = qubit_calibration_data.get("frequency_offset", 0.0)
        # Sample three different IY strength.
        old_beta = ix_params.get("beta", 0.0)
        if "drag_type" in ix_params:
            ix_params["drag_type"] = "01"
            default_delta_beta = 2.0
        elif "drag_type" not in ix_params:
            default_delta_beta = 100.0
        else:
            raise ValueError("Unknown drag type.")
        delta_beta = (
            old_beta
            if (old_beta > default_delta_beta and delta_beta is None)
            else delta_beta
        )
        delta_beta = default_delta_beta if delta_beta is None else delta_beta
        beta_list = np.array([0.0, -delta_beta, delta_beta]) + old_beta
        cr_params_lst.append(cr_params)
        ix_params_lst.append(ix_params)
        frequency_offset_lst.append(frequency_offset)
        beta_list_lst.append(beta_list)
        old_beta_lst.append(old_beta)

    ZZ_coeff_list_2D = np.zeros((3, len(qubit_pairs)))
    for _ in range(len(beta_list_lst[0])):
        beta_diff = []
        for n in range(len(qubit_pairs)):
            beta_diff.append(abs(beta_list_lst[n][_] - old_beta_lst[n]))
            ix_params_lst[n]["beta"] = beta_list_lst[n][_]
        if min(beta_diff) < 1.0e-6:
            _shots = shots * 2
        else:
            _shots = shots
        job_id = parallel_send_cr_tomography_job(
            qubits=qubit_pairs,
            backend=backend,
            cr_params_lst=cr_params_lst,
            ix_params_lst=ix_params_lst,
            cr_times=cr_times,
            frequency_offset_lst=frequency_offset_lst,
            shots=_shots,
            blocking=True,
        )
        coeff_dict_lst = process_parallel_tomo_data(
            job_id=job_id,
            show_plot=verbose,
        )
        ZZ_coeff_list_2D[_, :] = [coeff_dict["ZZ"] for coeff_dict in coeff_dict_lst]
        if max(beta_diff) < 1.0e5 and max(ZZ_coeff_list_2D[_, :]) < threshold_MHz:
            logger.info(
                f"ZZ error {round(ZZ_coeff_list_2D[_, :], 3)} MHz, no need for further calibration."
            )
            for i, qubit_pair in enumerate(qubit_pairs):
                qubit_calibration_data = read_calibration_data(backend, 
                                                               gate_name, 
                                                               tuple(qubit_pair))
                qubit_calibration_data.update(
                    {
                        "calibration_job_id": job_id,
                        "coeffs": coeff_dict_lst[i],
                    }
                )
                save_calibration_data(backend, gate_name, tuple(qubit_pair), qubit_calibration_data)
                if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
                    qubit_calibration_data = _update_frequency_offset(
                        qubit_calibration_data, "CR", backend.name
                    )
                return
        finished = []
        for i, qubit_pair in enumerate(qubit_pairs):
            if np.abs(ZZ_coeff_list_2D[_, i]) < threshold_MHz and beta_diff[i] < 1.0e5:
                logger.info("ZZ error for qubit pair " + \
                            f"{qubit_pair} is small enough.")     
                qubit_calibration_data = read_calibration_data(backend, 
                                                               gate_name, 
                                                               tuple(qubit_pair))
                qubit_calibration_data.update(
                    {
                        "calibration_job_id": job_id,
                        "coeffs": coeff_dict_lst[i],
                    }
                )
                save_calibration_data(backend, gate_name, 
                                      tuple(qubit_pair), 
                                      qubit_calibration_data)
                finished.append(qubit_pair)
                if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
                    qubit_calibration_data = _update_frequency_offset(
                        qubit_calibration_data, "CR", backend.name
                    )
        if len(finished) == len(qubit_pairs):
            return

    logger.info(f"ZZ sampling measurements complete : {ZZ_coeff_list_2D}." + "\n")
    # Fit a linear curve.
    fun = lambda x, a, b: a * x + b
    par_lst = []
    calibration_beta_lst = []
    for i in range(len(qubit_pairs)):
        par, _ = curve_fit(fun, beta_list_lst[i], ZZ_coeff_list_2D[:, i])
        par_lst.append(par)
        calibration_beta_lst.append(-par[1] / par[0])
        if verbose:
            fig, ax = plt.subplots(figsize=(4, 2), dpi=100)
            plt.scatter(beta_list_lst[i], ZZ_coeff_list_2D[:, i])
            x_line = np.linspace(min(beta_list_lst[i]), max(beta_list_lst[i]))
            y_line = fun(x_line, *par)
            plt.plot(x_line, y_line)
            plt.xlabel("beta")
            plt.ylabel("ZZ [MHz]")
            plt.show()
        ix_params_lst[i]["beta"] = calibration_beta_lst[i]
    logger.info(f"Calibrated IY beta: {calibration_beta_lst}" + "\n")

    # Perform a final tomography measurement.
    job_id = parallel_send_cr_tomography_job(
        qubits=qubit_pairs,
        backend=backend,
        cr_params_lst=cr_params_lst,
        ix_params_lst=ix_params_lst,
        cr_times=cr_times,
        frequency_offset_lst=frequency_offset_lst,
        blocking=True,
        shots=shots * 2,
    )

    coeff_dict_lst = process_parallel_tomo_data(job_id, show_plot=verbose)
    logger.info(f"Updated coupling strength: {coeff_dict_lst}")
    for i, qubit_pair in enumerate(qubit_pairs):
        qubit_calibration_data = read_calibration_data(backend, 
                                                       gate_name, 
                                                       tuple(qubit_pair))
        qubit_calibration_data.update(
            {
                "calibration_job_id": job_id,
                "coeffs": coeff_dict_lst[i],
                "ix_params": ix_params_lst[i],
            }
        )
        if np.abs(qubit_calibration_data["coeffs"]["IZ"]) > threshold_MHz:
            qubit_calibration_data = _update_frequency_offset(
                qubit_calibration_data, "CR", backend.name
            )

        save_calibration_data(backend, gate_name, tuple(qubit_pair), qubit_calibration_data)