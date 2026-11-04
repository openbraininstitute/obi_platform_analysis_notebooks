# ME-Model Custom Validation
Copyright (c) 2026 Open Brain Institute

Authors: Ilkan Kilic

last modified: 08.2026

## Summary
Demonstrates the OBI-One validation workflow with custom parametric validations on ME-Models.

This notebook uses:
- **BlueCelluLab's parametric validation framework** for composing custom tests
- **OBI-One's MEModelValidationWorkflow** for orchestration
- **entitysdk** for registering `ValidationResult` entities on the platform

## Adding Custom Validations

Each validation is a composition of:
1. **Protocol** — how to stimulate (e.g., `StepProtocol(threshold_percentage=150)`)
2. **Measurement** — what to extract (e.g., `EfelMeasurement(feature_name="Spikecount")`)
3. **Criterion** — pass/fail logic (e.g., `GreaterThan(threshold=3)`)

See section 6 of the notebook for the full guide.

## Use
1. Run cells sequentially
2. Authenticate when prompted
3. Select an ME-Model ID
4. Define your custom validations (or use the provided examples)
5. Run the workflow — results are automatically registered on the platform

## Requirements
- `bluecellulab` (with NEURON)
- `obi-one`
- `entitysdk`
- `obi-auth`
- `efel`
- `numpy`
- `matplotlib`
