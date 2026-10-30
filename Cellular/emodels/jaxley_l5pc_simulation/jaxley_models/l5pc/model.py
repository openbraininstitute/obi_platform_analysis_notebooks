# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os

import jaxley as jx
import numpy as np
from jaxley.channels import Leak

from jaxley_models.l5pc.channels import (
    SKE2,
    CaHVA,
    CaLVA,
    CaNernstReversal,
    CaPump,
    H,
    KPst,
    KTst,
    M,
    NaTaT,
    NaTs2T,
    SKv3_1,
)

params = {}
params["apical"] = {
    "apical_NaTs2T_gNaTs2T": 0.026145,
    "apical_SKv3_1_gSKv3_1": 0.004226,
    "apical_M_gM": 0.000143,
}
params["soma"] = {
    "somatic_NaTs2T_gNaTs2T": 0.983955,
    "somatic_SKv3_1_gSKv3_1": 0.303472,
    "somatic_SKE2_gSKE2": 0.008407,
    "somatic_CaPump_gamma": 0.000609,
    "somatic_CaPump_decay": 210.485291,
    "somatic_CaHVA_gCaHVA": 0.000994,
    "somatic_CaLVA_gCaLVA": 0.000333,
}
params["axon"] = {
    "axonal_NaTaT_gNaTaT": 3.137968,
    "axonal_KPst_gKPst": 0.973538,
    "axonal_KTst_gKTst": 0.089259,
    "axonal_SKE2_gSKE2": 0.007104,
    "axonal_SKv3_1_gSKv3_1": 1.021945,
    "axonal_CaHVA_gCaHVA": 0.00099,
    "axonal_CaLVA_gCaLVA": 0.008752,
    "axonal_CaPump_gamma": 0.00291,
    "axonal_CaPump_decay": 287.19873,
}
bounds = {
    "apical_NaTs2T_gNaTs2T": [0, 0.04],
    "apical_SKv3_1_gSKv3_1": [0, 0.04],
    "apical_M_gM": [0, 0.001],
    "somatic_NaTs2T_gNaTs2T": [0.0, 1.0],
    "somatic_SKv3_1_gSKv3_1": [0.25, 1],
    "somatic_SKE2_gSKE2": [0, 0.1],
    "somatic_CaPump_gamma": [0.0005, 0.01],
    "somatic_CaPump_decay": [20, 1_000],
    "somatic_CaHVA_gCaHVA": [0, 0.001],
    "somatic_CaLVA_gCaLVA": [0, 0.01],
    "axonal_NaTaT_gNaTaT": [0.0, 4.0],
    "axonal_KPst_gKPst": [0.0, 1.0],
    "axonal_KTst_gKTst": [0.0, 0.1],
    "axonal_SKE2_gSKE2": [0.0, 0.1],
    "axonal_SKv3_1_gSKv3_1": [0.0, 2.0],
    "axonal_CaHVA_gCaHVA": [0, 0.001],
    "axonal_CaLVA_gCaLVA": [0, 0.01],
    "axonal_CaPump_gamma": [0.0005, 0.05],
    "axonal_CaPump_decay": [20, 1_000],
}


def L5PC(ncomp=4, max_branch_len=300.0):
    base_path = os.path.dirname(__file__)
    cell = jx.read_swc(
        os.path.join(base_path, "morph_l5pc_with_axon.swc"),
        ncomp=ncomp,
        max_branch_len=max_branch_len,
        assign_groups=True,
    )

    ########## APICAL ##########
    # cell.apical.set("capacitance", 2.0)
    cell.apical.insert(NaTs2T().change_name("apical_NaTs2T"))
    cell.apical.insert(SKv3_1().change_name("apical_SKv3_1"))
    cell.apical.insert(M().change_name("apical_M"))
    cell.apical.insert(H().change_name("apical_H"))

    apical_inds = apical_inds = cell.nodes.loc[
        cell.nodes["apical"]
    ].global_branch_index.unique()
    for b in apical_inds:
        for comp in cell.branch(b).comps:
            distance = comp.distance(cell.branch(0).loc(0.0))
            cond = (-0.8696 + 2.087 * np.exp(distance * 0.0031)) * 8e-5
            comp.set("apical_H_gH", cond)

    ########## SOMA ##########
    cell.soma.insert(NaTs2T().change_name("somatic_NaTs2T"))
    cell.soma.insert(SKv3_1().change_name("somatic_SKv3_1"))
    cell.soma.insert(SKE2().change_name("somatic_SKE2"))
    ca_dynamics = CaNernstReversal()
    ca_dynamics.channel_constants["T"] = 307.15
    cell.soma.insert(ca_dynamics)
    cell.soma.insert(CaPump().change_name("somatic_CaPump"))
    cell.soma.insert(CaHVA().change_name("somatic_CaHVA"))
    cell.soma.insert(CaLVA().change_name("somatic_CaLVA"))
    cell.soma.set("CaCon_i", 5e-05)
    cell.soma.set("CaCon_e", 2.0)

    ########## BASAL ##########
    cell.basal.insert(H().change_name("basal_H"))
    cell.basal.set("basal_H_gH", 8e-5)

    # ########## AXON ##########
    cell.insert(CaNernstReversal())
    cell.set("CaCon_i", 5e-05)
    cell.set("CaCon_e", 2.0)

    cell.axon.insert(NaTaT().change_name("axonal_NaTaT"))
    cell.axon.insert(KTst().change_name("axonal_KTst"))
    cell.axon.insert(CaPump().change_name("axonal_CaPump"))
    cell.axon.insert(SKE2().change_name("axonal_SKE2"))
    cell.axon.insert(CaHVA().change_name("axonal_CaHVA"))
    cell.axon.insert(KPst().change_name("axonal_KPst"))
    cell.axon.insert(SKv3_1().change_name("axonal_SKv3_1"))
    cell.axon.insert(CaLVA().change_name("axonal_CaLVA"))

    ########## WHOLE CELL  ##########
    cell.insert(Leak())
    cell.set("Leak_gLeak", 3e-05)
    cell.set("Leak_eLeak", -75.0)

    cell.set("axial_resistivity", 100.0)
    cell.set("eNa", 50.0)
    cell.set("eK", -85.0)
    cell.set("v", -65.0)

    for group in ["apical", "soma", "axon"]:
        group_params = params[group]
        for key, value in group_params.items():
            cell.select(cell.nodes[cell.nodes[group]].index).set(key, value)

    return cell
