# Soft Matter Lab

An evolving collection of scripts for creating, running, and analysing soft-matter simulations. The repository is currently focused on Python analysis utilities; it is not yet a packaged, end-to-end workflow.

## Current repository structure

```text
.
├── analysis_scripts/
│   ├── soft_matter.py
│   ├── select_positions.py
│   ├── timeseries.py
│   └── equilibration_check.py
├── License.txt
└── README.md
```

`analysis_scripts/` currently contains the following pieces:

- `soft_matter.py` provides calculations for gyration tensors, radius of gyration, shape anisotropy, asphericity, end-to-end distance, and interfacial tension.
- `select_positions.py` selects polymer segments from a GSD system using MDAnalysis selections.
- `timeseries.py` applies selected calculations to every frame of a trajectory.
- `equilibration_check.py` is a configurable analysis script that reads `polymer_beads.gsd` and `thermodynamic_properties.h5` by default. It calculates selected structural and thermodynamic time series, estimates equilibration, and can write HDF5 outputs and plots.

## Using the scripts today

The scripts are intended to be copied or run from a simulation directory and adjusted for the system at hand. In particular, review the input and output filenames, bead-type selections, segment lengths, and enabled analyses near the top of `equilibration_check.py` before running it.

The current analysis scripts import NumPy, MDAnalysis, h5py, Matplotlib, and pymbar. They assume input data and conventions compatible with the paths and selections used in the code; examples, installation instructions, and a more complete description of those conventions are still to be added.

## Contributing

Contributions and improvements are welcome through pull requests. Since the repository is under active development, please check a script's assumptions and comments before relying on it for a new system.
