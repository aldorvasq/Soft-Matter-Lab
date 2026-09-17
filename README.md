# Soft Matter Lab

An evolving collection of scripts for creating, running, and analysing soft-matter simulations. The repository is currently focused on Python analysis utilities; it is not yet a packaged, end-to-end workflow.

## Current repository structure

```text
.
├── analysis_scripts/
│   ├── analysis_config.py
│   ├── soft_matter.py
│   ├── select_positions.py
│   ├── timeseries.py
│   └── equilibration_check.py
├── License.txt
└── README.md
```

`analysis_scripts/` currently contains the following pieces:

- `analysis_config.py` contains the user-editable inputs, analysis selections, system information, and output settings.
- `soft_matter.py` provides calculations for gyration tensors, radius of gyration, shape anisotropy, asphericity, end-to-end distance, and interfacial tension.
- `select_positions.py` selects polymer segments from a GSD system using MDAnalysis selections.
- `timeseries.py` applies selected calculations to every frame of a trajectory.
- `equilibration_check.py` runs the selected structural and thermodynamic analyses, estimates equilibration, and writes the requested results and plots.

## Running an analysis

Edit `analysis_scripts/analysis_config.py`; the calculation scripts should not normally need to be changed. Enable either or both analysis paths:

```python
run_structural = True
run_thermo = True
```

Then run:

```bash
python analysis_scripts/equilibration_check.py
```

Relative input and output paths are resolved from the directory in which this command is run.

### Thermodynamic input

The default configuration expects a HOOMD HDF5 log named `thermodynamic_properties.h5`. Leave the dataset mapping empty to use the built-in HOOMD paths:

```python
hdf5_input_file = 'thermodynamic_properties.h5'
hdf5_dataset_paths = {}
thermo_input_data = {}
```

For an HDF5 file with a different structure, provide a complete mapping from analysis names to dataset paths. A one-dimensional `timestep` dataset is required:

```python
hdf5_input_file = '/path/to/custom.h5'
hdf5_dataset_paths = {
    'timestep': '/simulation/time',
    'density_variance': '/analysis/density_var',
    'order_parameter': '/analysis/order_parameter',
}
thermo_input_data = {}

list_thermo = ['density_variance', 'order_parameter']
```

Preprocessed arrays can be supplied instead of an HDF5 file. Any number of one-dimensional quantities may be included, provided they have the same length as `timestep`:

```python
quantity_a = np.loadtxt('/path/to/quantity_a.txt').reshape(-1)
quantity_b = np.loadtxt('/path/to/quantity_b.txt').reshape(-1)

thermo_input_data = {
    'timestep': np.arange(quantity_a.size),
    'quantity_a': quantity_a,
    'quantity_b': quantity_b,
}

list_thermo = ['quantity_a', 'quantity_b']
```

Custom quantity names are used in the plot filenames. Preprocessed arrays and custom HDF5 mappings use `Frame` as the plot x-axis label; the built-in HOOMD input uses `Timestep`.

### Structural input

Structural analysis currently loads the configured trajectory with MDAnalysis:

```python
trajectory_file = 'polymer_beads.gsd'
list_bead_types = ['A', 'B', 'C', 'A B C']
list_length_of_polymer_segments = [5, 15, 60, 1260]
```

Each bead selection must have a matching segment length. Choose calculations through `list_analyses` and `list_timeseries`. If the trajectory does not contain usable box dimensions, provide them manually:

```python
box_dims = [48.0, 48.0, 48.0]
```

Leave `box_dims = None` to read the dimensions from the trajectory. Interfacial tension also requires box dimensions and a six-column `pressure_tensor` array.

### Outputs

When enabled, the workflow can create:

- `structural_analysis.h5` for structural results; it is not created during thermo-only runs.
- `interfacial_tension.h5` when interfacial tension can be calculated.
- One equilibration plot per selected quantity in `equilibration_plots/`.

Set `make_plots` and `save_raw_segments` in the configuration to control optional output.

## Current status

The current analysis scripts import NumPy, MDAnalysis, h5py, Matplotlib, and pymbar. They assume input data and conventions compatible with the paths and selections used in the code; examples, installation instructions, and a more complete description of those conventions are still to be added.

## Contributing

Contributions and improvements are welcome through pull requests. Since the repository is under active development, please check a script's assumptions and comments before relying on it for a new system.
