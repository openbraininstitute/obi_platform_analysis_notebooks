# This file is part of Jaxley-Models, a library of biophysical models for Jaxley.
# Jaxley-Models is licensed under the Apache License Version 2.0, see <https://www.apache.org/licenses/>

import os

import jaxley as jx
from jaxley.channels import HH


def CA1(ncomp=4, max_branch_len=300.0):
    base_path = os.path.dirname(__file__)
    cell = jx.read_swc(
        os.path.join(base_path, "morph_ca1_n120.swc"),
        ncomp=ncomp,
        max_branch_len=max_branch_len,
        assign_groups=True,
    )

    cell.insert(HH())

    cell.set("axial_resistivity", 1_000.0)
    cell.set("v", -62.0)
    cell.set("HH_m", 0.074901)
    cell.set("HH_h", 0.4889)
    cell.set("HH_n", 0.3644787)

    return cell
