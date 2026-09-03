import MDAnalysis as mda
import numpy as np
import soft_matter as sm


# Goal is to retrieve a list of indices that correspond to the polymer segments of interest, and then use those indices to select the positions of those segments from the trajectory.
# Then, we can compute the gyration tensor and its eigenvalues for each segment in each frame in a separate script.


def idx_selection(gsd_file, list_bead_types, list_length_of_polymer_segments):

    polymer_system = mda.Universe(gsd_file)
    polymer_system.trajectory[0] # Only valid for systems with constant N and invariant particle indices.

    # list_bead_types = ['A', 'B', 'C', 'all'] # you can also type two types ( 'A B' ) if you want to compute quantities for a specific combination of bead types.
    # list_length_of_polymer_segments = [10, 10, 100, 2100] # number of beads in a polymer segment, where segmet is backbone, sidechain, whole polymer, etc.

    if len(list_bead_types) != len(list_length_of_polymer_segments):
        raise ValueError("list_bead_types and list_length_of_polymer_segments must be the same length. Each bead type must have a corresponding polymer segment length.")

    # Each bead type gives a different shape, e.g. (2400, 10) for 'A' but
    # (24, 2100) for 'all', so a single ndarray cannot hold them all. Key the
    # index arrays by bead type instead.
    master_idx_dic = {}

    for bead_type, length_of_polymer_segment in zip(list_bead_types, list_length_of_polymer_segments):
        if type(bead_type) != str or bead_type == '': # Check for empty string or non-string bead type
            raise ValueError("bead_type must be a string, e.g. 'C' or 'all' or 'A B'.")
        if bead_type != 'all':
            bead_type_selection = f'type {bead_type}'
            print(f"Selecting beads of type {bead_type} with selection string '{bead_type_selection}'.")
        else:
            print("Either 'all' was specified as the bead type, or some other string was provided. All beads will be selected.")
            bead_type_selection = 'all'

        # Not needed but keeping for clarity. The length of the polymer segment is already defined in the list above.
        length_of_polymer_segment = length_of_polymer_segment # number of beads in a polymer segment (backbone, sidechain, whole polymer, etc.)

        # How many chains, backbones, sidechains, etc. are in the system. This is used to define the shape of the positions array below.
        # Assumption is that each segment is equally sized.
        n_beads_of_type = polymer_system.select_atoms(bead_type_selection).n_atoms
        if n_beads_of_type % length_of_polymer_segment != 0:
            raise ValueError(f"Number of beads of type {bead_type} ({n_beads_of_type}) is not divisible by the length of the polymer segment ({length_of_polymer_segment}).")
        else:
            number_of_segments = n_beads_of_type // length_of_polymer_segment


        select_beads = polymer_system.select_atoms(bead_type_selection)

        # .indices are the atom indices in the Universe, which is what the
        # trajectory loop needs to slice positions with. np.arange would instead
        # give positions within the selection, which are only the same thing when
        # the selection happens to start at atom 0.
        # The reshape assumes each segment's beads are contiguous in the selection.
        idx_array = select_beads.indices.reshape(number_of_segments,
                                                 length_of_polymer_segment)

        # Store the indices in the master dictionary, keyed by bead type.
        master_idx_dic[bead_type] = idx_array

    return master_idx_dic

# print(idx_selection('polymer_beads.gsd'))
