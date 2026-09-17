"""User-editable settings for equilibration_check.py."""

import numpy as np


# ---- input data -----------------------------------------------------------------------

hdf5_input_file = 'thermodynamic_properties.h5'
trajectory_file = 'polymer_beads.gsd'

# Leave empty to use the built-in HOOMD dataset paths. For a different HDF5 layout,
# provide a complete mapping of analysis names to dataset paths here instead.
hdf5_dataset_paths = {}


# Leave this dictionary empty to read datasets from hdf5_input_file. To bypass HDF5,
# load or prepare arrays here instead. Every series must have the same length as
# timestep.
thermo_input_data = {}

# Any number of preprocessed quantities can be supplied. Each dictionary key becomes
# its analysis name, and all arrays MUST have the same length as timestep:
# quantity_A = np.loadtxt('/path/to/quantity_A.txt').reshape(-1)
# quantity_B = np.loadtxt('/path/to/quantity_B.txt').reshape(-1)
# quantity_C = np.loadtxt('/path/to/quantity_C.txt').reshape(-1)
# thermo_input_data = {
#     'timestep': np.arange(quantity_A.size),
#     'quantity_A': quantity_A,
#     'quantity_B': quantity_B,
#     'quantity_C': quantity_C,
# }


# ---- what to run ----------------------------------------------------------------------

run_structural = True
run_thermo = True

# Names here may be built-in names or any custom keys from hdf5_dataset_paths or
# thermo_input_data.

# Built-in options: 'potential_energy', 'pressure', 'interfacial_tension',
# 'pressure_tensor_xx', 'pressure_tensor_xy', 'pressure_tensor_xz',
# 'pressure_tensor_yy', 'pressure_tensor_yz', and 'pressure_tensor_zz'.
list_thermo = ['potential_energy', 'pressure', 'interfacial_tension',
               'pressure_tensor_xx', 'pressure_tensor_yy', 'pressure_tensor_zz']

# Built-in options: 'rg_components', 'shape_anisotropy', and 'asphericity'.
list_analyses = ['rg_components', 'shape_anisotropy', 'asphericity']
list_timeseries = ['end_to_end']


# ---- structural system ---------------------------------------------------------------

# Optional manual box lengths [Lx, Ly, Lz]. Leave as None to read them from the
# trajectory. This can also supply the box for thermo-only interfacial tension.
box_dims = None

# Each bead selection must have a corresponding segment length.
list_bead_types = ['A', 'B', 'C', 'A B C']
list_length_of_polymer_segments = [5, 15, 60, 1260]

# Per-case scalar metadata. Each function receives a context dictionary containing
# box, indices, n_frames, and case.
user_quantities = {
    'surface_concentration':
        lambda context: (context['indices']['C'].size
                         / (2.0 * context['box'][0] * context['box'][1])),
}


# ---- output ---------------------------------------------------------------------------

output_file = 'structural_analysis.h5'
gamma_output_file = 'interfacial_tension.h5'
plot_folder = 'equilibration_plots'
save_raw_segments = False
make_plots = True


# ---- equilibration detection ----------------------------------------------------------

fix_uncut_transients = True
start_offset_threshold = 5.0
