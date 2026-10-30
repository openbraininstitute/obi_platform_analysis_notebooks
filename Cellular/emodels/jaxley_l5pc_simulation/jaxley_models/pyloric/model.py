# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

from io import StringIO

import jax.numpy as jnp
import jaxley as jx
import pandas as pd

from jaxley_models.pyloric.channels import (
    A,
    CaNernstReversal,
    CaS,
    CaT,
    H,
    KCa,
    Kd,
    Leak,
    Na,
)
from jaxley_models.pyloric.synapses import CholinergicSynapse, GlutamatergicSynapse

prinz04_params = """
# Parameters for the Pyloric network (Prinz et al. 2004, Tab. 2)
model_neuron,gNa,gCaT,gCaS,gA,gKCa,gKd,gH,gLeak
ab_pd,0.400,0.0025,0.006,0.050,0.010,0.100,0.00001,0.00000
ab_pd,0.100,0.0025,0.004,0.050,0.005,0.100,0.00001,0.00000
ab_pd,0.200,0.0025,0.004,0.050,0.005,0.050,0.00001,0.00000
ab_pd,0.200,0.0050,0.004,0.040,0.005,0.125,0.00001,0.00000
ab_pd,0.300,0.0025,0.002,0.010,0.005,0.125,0.00001,0.00000
lp,0.100,0.0000,0.008,0.040,0.005,0.075,0.00005,0.00002
lp,0.100,0.0000,0.006,0.030,0.005,0.050,0.00005,0.00002
lp,0.100,0.0000,0.010,0.050,0.005,0.100,0.00000,0.00003
lp,0.100,0.0000,0.004,0.020,0.000,0.025,0.00005,0.00003
lp,0.100,0.0000,0.004,0.030,0.000,0.075,0.00005,0.00002
py,0.100,0.0025,0.002,0.050,0.000,0.125,0.00005,0.00001
py,0.200,0.0075,0.000,0.050,0.000,0.075,0.00005,0.00000
py,0.200,0.0100,0.000,0.050,0.000,0.100,0.00005,0.00000
py,0.400,0.0025,0.002,0.050,0.000,0.075,0.00005,0.00000
py,0.500,0.0025,0.002,0.040,0.000,0.125,0.00001,0.00003
py,0.500,0.0025,0.002,0.040,0.000,0.125,0.00000,0.00002
"""

pyloric_params_bounds = {
    # taken from https://github.com/mackelab/pyloric
    # === Neuron maximal conductances (mS/cm^2) ===
    # AB/PD neuron
    "ab_pd_Na_gNa": (0.0, 1000.0),
    "ab_pd_CaT_gCaT": (0.0, 10.0),
    "ab_pd_CaS_gCaS": (0.0, 20.0),
    "ab_pd_A_gA": (0.0, 100.0),
    "ab_pd_KCa_gKCa": (0.0, 200.0),
    "ab_pd_Kd_gKd": (0.0, 200.0),
    "ab_pd_H_gH": (0.0, 0.2),
    "ab_pd_Leak_gLeak": (0.0, 0.1),
    # LP neuron
    "lp_Na_gNa": (0.0, 1000.0),
    "lp_CaT_gCaT": (0.0, 0.0),
    "lp_CaS_gCaS": (0.0, 20.0),
    "lp_A_gA": (0.0, 100.0),
    "lp_KCa_gKCa": (0.0, 200.0),
    "lp_Kd_gKd": (0.0, 200.0),
    "lp_H_gH": (0.0, 0.2),
    "lp_Leak_gLeak": (0.0, 0.1),
    # PY neuron
    "py_Na_gNa": (0.0, 1000.0),
    "py_CaT_gCaT": (0.0, 10.0),
    "py_CaS_gCaS": (0.0, 0.0),
    "py_A_gA": (0.0, 100.0),
    "py_KCa_gKCa": (0.0, 0.0),
    "py_Kd_gKd": (0.0, 200.0),
    "py_H_gH": (0.0, 0.2),
    "py_Leak_gLeak": (0.0, 0.1),
    # === Synaptic maximal conductances (nS) ===
    "GlutamatergicSynapse_gS": (0.0, 1000.0),
    "CholinergicSynapse_gS": (0.0, 1000.0),
    # === Synaptic reversal potentials (mV) ===
    "CholinergicSynapse_e_syn": (-80.0, -70.0),
    "GlutamatergicSynapse_e_syn": (-80.0, -70.0),
}


def PyloricNetwork() -> jx.Network:
    """Model of the pyloric circuit.

    Model of the pyloric circuit according to:

    Prinz et al. 2003, J Neurophysiol

    We model the network using three cells using the `jx.Network` class:
    - ab_pd: abducens and pre-dorsal pyloric neuron
    - lp: lateral pyloric neuron
    - py: posteroventral pyloric neuron
    which can be accessed by the group names "ab_pd", "lp", and "py" respectively.

    The network is connected as follows using glutamatergic and cholinergic synapses:
    - ab_pd -> lp : glutamatergic
    - ab_pd -> lp : cholinergic
    - ab_pd -> py : glutamatergic
    - ab_pd -> py : cholinergic
    - lp -> ab_pd : glutamatergic
    - lp -> py : glutamatergic
    - py -> lp : glutamatergic

    The simulator was re-implemented based on the cython implementation from:
    https://github.com/mackelab/pyloric and matches the results from the original code.

    Returns:
        The pyloric circuit network.
    """
    # build network
    net = jx.Network([jx.Cell()] * 3)
    net.cell(0).add_to_group("ab_pd")
    net.cell(1).add_to_group("lp")
    net.cell(2).add_to_group("py")

    # insert channels
    net.insert(CaNernstReversal())
    net.insert(Na())
    net.insert(CaT())
    net.insert(CaS())
    net.insert(A())
    net.insert(KCa())
    net.insert(Kd())
    net.insert(H())
    net.insert(Leak())

    # set cell parameters
    area = 0.6283 * 1e-3  # cm²
    C = 0.628 * 1e-3  # uF
    radius = jnp.sqrt(area / (2 * jnp.pi)) * 1e4  # um
    net.set("capacitance", C / area)  # uF / cm²
    net.set("length", radius)
    net.set("radius", radius)
    net.set("v", -50.0)

    params = pd.read_csv(StringIO(prinz04_params), comment="#")
    params = params.groupby("model_neuron").last().T.to_dict()
    for neuron, conductances in params.items():
        for conductance, value in conductances.items():
            net.__getattr__(neuron).set(f"{conductance[1:]}_{conductance}", value)

    # Create synaptic connections
    glut = GlutamatergicSynapse()
    chol = CholinergicSynapse()
    jx.connect(net.ab_pd, net.lp, glut)  # AB-LP
    jx.connect(net.ab_pd, net.lp, chol)  # PD-LP

    jx.connect(net.ab_pd, net.py, glut)  # AB-PY
    jx.connect(net.ab_pd, net.py, chol)  # PD-PY

    jx.connect(net.lp, net.ab_pd, glut)  # LP-PD
    jx.connect(net.lp, net.py, glut)  # LP-PY

    jx.connect(net.py, net.lp, glut)  # PY-LP
    return net
