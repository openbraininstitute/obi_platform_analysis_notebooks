# Circuit registration

Copyright (c) 2026 Open Brain Institute

Authors: Christoph Pokorny

Last modified: 06.2026

## Summary
This notebook allows a user to register a new SONATA circuit together with metadata as a private `Circuit` entity in the entitycore database. Additional circuit assets (connectivity matrix, basic connectivity plots, etc.) will be generated automatically upon upload.

## Use
The SONATA circuit should be provided by compressing the whole circuit folder into a zip file and copying it into the JupyterHub workspace. All required metadata can be entered diretly within the notebook. Where applicable, interactive widgets can be used to select metadata options and linked entities in a user-friendly way. All required linked entities (publications, contributors, subjects, etc.) must exist already in the database, i.e., they have to be registered separately beforehand.

Optionally, the main overview and simulation designer images can be provided as well; otherwise default images will be generated.

**Important:** [SNAP circuit validation](https://github.com/openbraininstitute/snap#circuit-validation) needs to pass for a new circuit to be registered.
