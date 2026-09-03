import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pymbar import timeseries
import soft_matter as sm
import MDAnalysis as mda
from select_positions import idx_selection
from timeseries import gtensor_timeseries, e2e_timeseries


# -------------------------------------------------------------------------------------
# STEP 1. Read the potential energy and the timestep out of the h5 file.
# Same access pattern as Scripts/plot_quantities.py: open the file, then index the
# dataset by its full path inside the file and take [:] to pull it into a numpy array.
# -------------------------------------------------------------------------------------

hdf5_input_file = 'thermodynamic_properties.h5'
input_file = 'polymer_beads.gsd'  # GSD file containing the polymer system

# ---- output settings, all in one place ------------------------------------------------
output_file = 'structural_analysis.h5'   # everything the analysis produces
gamma_output_file = 'interfacial_tension.h5'
plot_folder = 'equilibration_plots'
save_raw_segments = False   # the per-segment arrays are the bulk of the file
make_plots = True

# ---- what to run ----------------------------------------------------------------------
run_structural = True    # the gyration tensor analyses, per bead type
run_thermo = True        # the quantities logged by HOOMD, on their own clock

# When detect_equilibration leaves t0 at the very start of a series that clearly has a
# transient, fall back to cutting at the point the series first reaches its plateau.
# See the note next to the fallback itself for why the heuristic fails this way.
fix_uncut_transients = True
start_offset_threshold = 5.0   # in units of the within-frame spread

hdf5_file = h5py.File(name=hdf5_input_file, mode='r')
timestep = hdf5_file['hoomd-data/Simulation/timestep'][:]
potet = hdf5_file['/hoomd-data/md/compute/ThermodynamicQuantities/potential_energy'][:]
pressure = hdf5_file['/hoomd-data/md/compute/ThermodynamicQuantities/pressure'][:]
pressure_tensor = hdf5_file['/hoomd-data/md/compute/ThermodynamicQuantities/pressure_tensor'][:]
hdf5_file.close()

#  Polymer structural quantities
polymer_system = mda.Universe(input_file) # Retrieve system from GSD file.
box_dims = polymer_system.dimensions[0:3] # box dimensions in x, y, z
n_frames = len(polymer_system.trajectory) # number of frames in the trajectory


# # List of bead types to compute quantities for. Can be any combination of the bead types in the system, or just one type, or 'all' for all beads.
list_bead_types = ['A', 'B', 'C', 'A B C'] # you can also type two types ( 'A B' ) if you want to compute quantities for a specific combination of bead types.

list_length_of_polymer_segments = [5, 15, 60, 1260] # number of beads in a polymer segment, where segmet is backbone, sidechain, whole polymer, etc.

# Registry of the analyses that can be run. Each entry says which function does the
# work, what that function needs as input, and what it gives back.
#
#   function : the callable, taken from soft_matter.py
#   needs    : the inputs, in the order the function takes them. A name listed in
#              per_segment_inputs below is sliced down to one segment of one frame;
#              anything else is passed through whole (e.g. the box).
#   returns  : the output column name(s). A function returning several values, like
#              radius_of_gyration_components, lists them all here.
#
# Adding a new analysis is one entry here plus, if it needs an input that is not
# already available, one line in per_segment_inputs. The loop below does not change.
analysis_registry = {
    'rg_components': {'function': sm.radius_of_gyration_components,
                      'needs': ('eigenvalues', 'eigenvectors'),
                      'returns': ('rg_total', 'rg_parallel', 'rg_perpendicular')},
    'shape_anisotropy': {'function': sm.shape_anisotropy,
                         'needs': ('eigenvalues',),
                         'returns': ('shape_anisotropy',)},
    'asphericity': {'function': sm.asphericity,
                    'needs': ('eigenvalues',),
                    'returns': ('asphericity',)},
}

# Which of the above to actually run. Drop a name to skip that analysis.
list_analyses = ['rg_components', 'shape_anisotropy', 'asphericity']

# Inputs that do not vary per segment, passed to the analysis functions unchanged.
constant_inputs = {'box': box_dims}

# A second kind of analysis, which walks the trajectory itself and hands back a whole
# (n_frames, n_segments) column at once rather than one value per segment. These do
# not fit the per-segment registry above, so they are called once up front and their
# column is merged into the results below.
timeseries_registry = {'end_to_end': e2e_timeseries}

# Which of those to run. Drop a name to skip its trajectory pass entirely.
list_timeseries = ['end_to_end']

# Per-case scalars: one number describing this simulation, not a timeseries. They are
# written as root attributes of the output file so a later script comparing cases can
# read them without parsing directory names.
#
# These live here rather than in soft_matter.py on purpose. soft_matter holds general
# physics that applies to any CG polymer system; how a particular sweep defines its
# control parameter is specific to this project's directory layout and conventions, and
# would not mean anything to someone else using the module.
#
# Each entry is a function of `context`, a dictionary assembled below holding box,
# indices, n_frames and the case directory. Edit or add freely.
user_quantities = {
    # Backbone beads per unit interfacial area. The C selection is the backbone, so its
    # segment count times its segment length is the total number of backbone beads, and
    # a slab in a periodic box presents two interfaces of area Lx * Ly.
    'surface_concentration':
        lambda context: (context['indices']['C'].size
                         / (2.0 * context['box'][0] * context['box'][1])),
}

indices = idx_selection(input_file, list_bead_types, list_length_of_polymer_segments) # Retrieve indices of polymer segments of interest from select_positions.py

# The per-case scalars, evaluated now that indices and the box are known.
context = {'box': box_dims, 'indices': indices, 'n_frames': n_frames,
           'case': os.path.basename(os.path.abspath('.'))}
scalars = {name: float(function(context)) for name, function in user_quantities.items()}
for name, value in scalars.items():
    print(f"  {name} = {value:.6g}")

eigs, vecs, steps = gtensor_timeseries(polymer_system, indices, box_dims) # Retrieve gyration tensor eigenvalues for each segment of interest from timeseries.py

# Each of these does its own pass over the trajectory, so only the requested ones run.
# The steps are discarded because gtensor_timeseries already returned the same array.
precomputed = {}
for name in list_timeseries:
    precomputed[name], _ = timeseries_registry[name](polymer_system, indices, box_dims)

results = {} # Raw per-segment data: results[bead_type][column] is (n_frames, n_segments).
series = {}  # Segment-averaged data: series[bead_type][column] is (n_frames,).
spread = {}  # Standard error across segments within each frame, same shape as series.

for bead_type in eigs:
    eig = eigs[bead_type]                     # (n_frames, n_segments, 3)
    vec = vecs[bead_type]                     # (n_frames, n_segments, 3, 3)
    number_of_segments = eig.shape[1]         # read off, no longer computed

    # Inputs that vary per segment. An analysis asking for one of these gets the
    # single segment's slice, e.g. a (3,) array of eigenvalues. Adding a new kind
    # of input, such as positions, means adding one line here.
    per_segment_inputs = {'eigenvalues': eig, 'eigenvectors': vec}

    # One output array per column produced by the selected analyses. A single
    # analysis can produce several columns, which is why this walks 'returns'.
    output_names = [column for name in list_analyses
                    for column in analysis_registry[name]['returns']]
    quantities = {column: np.zeros((n_frames, number_of_segments))
                  for column in output_names}

    for ts in range(n_frames):
        for segment in range(number_of_segments):
            for name in list_analyses:
                analysis = analysis_registry[name]

                # Build the argument list in the order the function expects, taking
                # per-segment inputs sliced down to this frame and segment, and
                # everything else whole.
                args = [per_segment_inputs[key][ts, segment]
                        if key in per_segment_inputs else constant_inputs[key]
                        for key in analysis['needs']]

                values = analysis['function'](*args)
                # Functions returning a single value are wrapped so that one code
                # path handles both cases.
                if len(analysis['returns']) == 1:
                    values = (values,)

                for column, value in zip(analysis['returns'], values):
                    quantities[column][ts, segment] = value

    # Merge in the columns that were computed over the whole trajectory. They are
    # already (n_frames, n_segments), so they slot straight in alongside the others.
    for name in list_timeseries:
        quantities[name] = precomputed[name][bead_type]

    # Store per bead type, otherwise each pass overwrites the last.
    results[bead_type] = quantities

    # Collapse the segment axis to get one number per frame, which is the form the
    # equilibration analysis consumes. The raw per-segment arrays are kept untouched
    # in results, so nothing is lost by averaging here.
    series[bead_type] = {column: arr.mean(axis=1)
                         for column, arr in quantities.items()}

    # How much the segments disagree within a single frame, as the standard error of
    # that frame's mean. This is the instantaneous sampling error, and it is the
    # scale that any drift in the series has to be judged against: a trend is only
    # meaningful if it is large compared to this.
    spread[bead_type] = {column: arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])
                         for column, arr in quantities.items()}

    summary = '  '.join(f"{column} {series[bead_type][column].mean():.3f}"
                        for column in quantities)
    print(f"{bead_type:>5}: {number_of_segments:>5} segments |  {summary}")


# -------------------------------------------------------------------------------------
# Equilibration detection, one pass per segment-averaged series.
#
# detect_equilibration takes a 1-D series and returns three numbers:
#   t0   -> index where the production region starts, everything before it is burn-in
#   g    -> statistical inefficiency, how many samples apart two points must sit to
#           count as independent
#   Neff -> how many independent samples the production region is worth
#
# It is a search for the cut point that leaves the most independent samples, not a
# test, so it always returns a value. A t0 that ate most of the run means the series
# never settled rather than that equilibration finished late.
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# Thermodynamic quantities, kept separate from the structural ones because they sit on
# a different clock: HDF5Log saves every 500 steps, the trajectory every 30,000, so
# there are 30,000 thermodynamic samples against 500 frames. They also have no segment
# axis, so there is no spread across segments for them.
#
# The interfacial tension is derived here rather than logged by HOOMD. It is computed
# on the whole timeseries in one call, since soft_matter.interfacial_tension is
# vectorised over the leading axis.
# -------------------------------------------------------------------------------------

gamma = sm.interfacial_tension(pressure_tensor, box_dims,
                               normal_axis=2,      # interfaces are normal to z
                               n_interfaces=2)     # a slab in a periodic box has two

# HOOMD stores the pressure tensor as six components in this order.
tensor_component_names = ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')

# Registry of thermodynamic quantities, selectable the same way the structural ones
# are. Each entry is just the series itself, since these are already computed rather
# than derived per segment.
thermo_registry = {'potential_energy': potet,
                   'pressure': pressure,
                   'interfacial_tension': gamma}
for position, name in enumerate(tensor_component_names):
    thermo_registry[f'pressure_tensor_{name}'] = pressure_tensor[:, position]

# Which of the above to analyse. Drop a name to skip it.
list_thermo = ['potential_energy', 'pressure', 'interfacial_tension',
               'pressure_tensor_xx', 'pressure_tensor_yy', 'pressure_tensor_zz']

thermo_series = {name: thermo_registry[name] for name in list_thermo}


# -------------------------------------------------------------------------------------
# Write the interfacial tension to its own file rather than back into the simulation's
# log. thermodynamic_properties.h5 is raw simulation output and cannot be regenerated
# without re-running the simulation, so nothing derived is written into it.
#
# The file carries its own copy of the timestep axis, so it stands alone: reading gamma
# does not require opening the thermo log to find out when each sample was taken. The
# parameters that define gamma are stored as attributes, since it is meaningless
# without knowing which axis was taken as the normal and how many interfaces the box
# holds.
#
#   /timestep              (n_thermo,)   MD timestep of each sample
#   /interfacial_tension   (n_thermo,)   gamma, with the convention in its attributes
# -------------------------------------------------------------------------------------

with h5py.File(gamma_output_file, 'w') as gamma_file:
    gamma_file.attrs['source_log'] = hdf5_input_file
    gamma_file.attrs['formula'] = '(L_n / n_interfaces) * (P_nn - (P_t1t1 + P_t2t2) / 2)'
    gamma_file.attrs['normal_axis'] = 2
    gamma_file.attrs['n_interfaces'] = 2
    gamma_file.attrs['box'] = box_dims
    gamma_file.attrs['source'] = 'soft_matter.interfacial_tension, via equilibration_check.py'

    gamma_file.create_dataset('timestep', data=timestep)
    gamma_file.create_dataset('interfacial_tension', data=gamma)

print(f"wrote {gamma_output_file}")



# -------------------------------------------------------------------------------------
# detect_equilibration maximises Neff = (N - t) / g, and g comes from a correlation
# function normalised by the series variance. A short, sharp transient at the start
# inflates that variance, which deflates g and lets t = 0 win the search: cutting the
# transient then makes g worse rather than better, so the cut point never moves off
# zero even though the start is plainly not equilibrated.
#
# The fallback below is the test the eye already applies. Take the second half of the
# series as a reference plateau, find the first point that comes within a few spreads
# of it, and refuse to start the production region before that. Then recompute g and
# Neff on what is left. It is a heuristic on top of a heuristic, so the plots should
# still be looked at, but it fails in the safe direction: it can only ever move t0
# later, never earlier.
# -------------------------------------------------------------------------------------

def transient_end(one_series, scale, tolerance=3.0):
    """First index at which the series has reached its own late-time plateau."""
    reference = np.median(one_series[len(one_series) // 2:])
    if not scale > 0:
        return 0
    within = np.abs(one_series - reference) < tolerance * scale
    if not within.any():
        return 0
    return int(np.argmax(within))


def analyse_series(one_series, scale, nskip=1):
    """detect_equilibration, with the start-of-run fallback applied when needed.

    Returns t0, g, Neff, and how far the first retained point sits from the production
    mean in units of `scale`, which is the diagnostic that triggers the fallback.
    """
    t0, g, neff = timeseries.detect_equilibration(one_series, nskip=nskip)
    t0 = int(t0)

    def offset(cut):
        mean = one_series[cut:].mean()
        return abs(one_series[cut] - mean) / scale if scale > 0 else 0.0

    start_offset = offset(t0)

    if fix_uncut_transients and start_offset > start_offset_threshold:
        floor = transient_end(one_series, scale)
        if floor > t0:
            t0 = floor
            # Recompute the correlation on the trimmed series: g measured across the
            # transient is not the g of the equilibrated region.
            g = float(timeseries.statistical_inefficiency(one_series[t0:], fast=True))
            neff = float((len(one_series) - t0 + 1) / g)
            start_offset = offset(t0)

    return t0, float(g), float(neff), float(start_offset)


equilibration = {}  # equilibration[bead_type][column] = {'t0':..., 'g':..., 'neff':...}

sampling_period = steps[1] - steps[0]  # MD steps between trajectory frames

print()
print(f"{'bead type':>9} {'quantity':>17} {'t0':>5} {'t0 (steps)':>12} "
      f"{'discarded':>10} {'g':>8} {'g (steps)':>11} {'Neff':>9} "
      f"{'production mean':>24}")
print('-' * 118)

for bead_type in series:
    equilibration[bead_type] = {}
    for column, one_series in series[bead_type].items():
        t0, g, neff, start_offset = analyse_series(
            one_series, np.median(spread[bead_type][column]))

        # The production average and its uncertainty. The error uses Neff rather than
        # the number of frames, because consecutive frames are correlated: dividing by
        # sqrt(n_frames) would understate the error by sqrt(g), which for the slower
        # quantities here is a factor of three or more.
        production = one_series[t0:]
        production_mean = production.mean()
        production_error = production.std(ddof=1) / np.sqrt(neff)


        equilibration[bead_type][column] = {'t0': int(t0),
                                            'g': float(g),
                                            'neff': float(neff),
                                            'production_mean': float(production_mean),
                                            'production_error': float(production_error),
                                            'start_offset': float(start_offset)}

        print(f"{bead_type:>9} {column:>17} {int(t0):>5} {steps[t0]:>12,} "
              f"{100 * t0 / n_frames:>9.1f}% {g:>8.2f} {g * sampling_period:>11,.0f} "
              f"{neff:>9.1f} {production_mean:>13.5g} +/- {production_error:<8.3g}")

# The same treatment for the thermodynamic series. detect_equilibration tests every
# index as a candidate cut point by default, which is slow on 30,000 samples, so this
# sparsifies the search to every 100th index. That costs a little resolution in t0 and
# nothing in accuracy.
thermo_nskip = 100
thermo_sampling_period = timestep[1] - timestep[0]
n_thermo = len(timestep)

equilibration['thermo'] = {}

for column, one_series in (thermo_series.items() if run_thermo else []):
    # No spread across segments exists for these, so the production standard deviation
    # stands in as the scale of a normal fluctuation.
    t0, g, neff, start_offset = analyse_series(
        one_series, one_series[len(one_series) // 2:].std(ddof=1), nskip=thermo_nskip)

    production = one_series[t0:]
    production_mean = production.mean()
    production_error = production.std(ddof=1) / np.sqrt(neff)


    equilibration['thermo'][column] = {'t0': int(t0),
                                       'g': float(g),
                                       'neff': float(neff),
                                       'production_mean': float(production_mean),
                                       'production_error': float(production_error),
                                       'start_offset': float(start_offset)}

    print(f"{'thermo':>9} {column:>17} {int(t0):>5} {timestep[t0]:>12,} "
          f"{100 * t0 / n_thermo:>9.1f}% {g:>8.2f} {g * thermo_sampling_period:>11,.0f} "
          f"{neff:>9.1f} {production_mean:>13.5g} +/- {production_error:<8.3g}")


# -------------------------------------------------------------------------------------
# Write everything to an HDF5 file, laid out as paths in the same spirit as HOOMD's
# own log file, so a later script can pull out one quantity without recomputing.
#
#   /steps                              (n_frames,)   MD timestep of each frame
#   /series/<bead_type>/<quantity>      (n_frames,)   segment-averaged, one per frame
#   /spread/<bead_type>/<quantity>      (n_frames,)   standard error across segments
#   /segments/<bead_type>/<quantity>    (n_frames, n_segments)   raw, per segment
#
# The equilibration numbers ride along as attributes on the matching /series dataset,
# so t0, g and Neff can never drift apart from the data they describe:
#
#   f['series/C/rg_total'][:]            -> the series
#   f['series/C/rg_total'].attrs['t0']   -> where equilibration ends, as a frame index
#
# So plotting Rg against asphericity for the backbones is just two reads from
# /series/C/, and the frame axis is shared with /steps.
# -------------------------------------------------------------------------------------


with h5py.File(output_file, 'w') as out:
    # Enough context to know what produced this file without going back to the script.
    out.attrs['source_trajectory'] = 'polymer_beads.gsd'
    out.attrs['n_frames'] = n_frames
    out.attrs['sampling_period'] = sampling_period
    out.attrs['bead_types'] = list_bead_types
    out.attrs['segment_lengths'] = list_length_of_polymer_segments

    # Per-case scalars from user_quantities, one value describing this simulation.
    for name, value in scalars.items():
        out.attrs[name] = value

    out.create_dataset('steps', data=steps)

    # Thermodynamic series live on their own clock, so they carry their own steps axis
    # under the same group rather than sharing /steps.
    if run_thermo:
        out.create_dataset('thermo/timestep', data=timestep)
        for column, one_series in thermo_series.items():
            dataset = out.create_dataset(f'thermo/{column}', data=one_series)
            for key, value in equilibration['thermo'][column].items():
                dataset.attrs[key] = value

    for bead_type in series:
        out.create_group(f'series/{bead_type}')
        for column, one_series in series[bead_type].items():
            dataset = out.create_dataset(f'series/{bead_type}/{column}', data=one_series)
            # Equilibration results attached to the series they were computed from.
            for key, value in equilibration[bead_type][column].items():
                dataset.attrs[key] = value

            out.create_dataset(f'spread/{bead_type}/{column}',
                               data=spread[bead_type][column])

            if save_raw_segments:
                # Compressed because these are the large arrays, e.g. 500 x 2400
                # for every sidechain quantity.
                out.create_dataset(f'segments/{bead_type}/{column}',
                                   data=results[bead_type][column],
                                   compression='gzip', compression_opts=4)

print()
print(f"wrote {output_file}")


# -------------------------------------------------------------------------------------
# One figure per bead type per quantity, showing what the equilibration numbers mean:
#
#   grey line     - the segment-averaged series
#   grey band     - the within-frame spread, i.e. how much the segments disagree at a
#                   single instant. Any drift has to be judged against this band, not
#                   against the size of the initial transient, which is what makes a
#                   still-drifting series look flat by eye.
#   red line      - t0, where equilibration is judged to end
#   green shading - the production region that survives the cut
#   green dots    - the effectively independent samples, spaced about g apart
#   blue dashes   - the mean over those independent samples
# -------------------------------------------------------------------------------------

if make_plots:
    os.makedirs(plot_folder, exist_ok=True)

    # Structural and thermodynamic series are plotted by the same code. They differ only
    # in their steps axis and in whether a within-frame spread exists, so both are
    # gathered into one list of (group, column, series, steps, spread) and drawn once.
    to_plot = []
    for bead_type in series:
        for column, one_series in series[bead_type].items():
            to_plot.append((bead_type, column, one_series, steps,
                            spread[bead_type][column]))
    if run_thermo:
        for column, one_series in thermo_series.items():
            to_plot.append(('thermo', column, one_series, timestep, None))

    for bead_type, column, one_series, x_axis, error in to_plot:
            t0 = equilibration[bead_type][column]['t0']
            g = equilibration[bead_type][column]['g']
            neff = equilibration[bead_type][column]['neff']

            # The samples the production average is actually built from: everything
            # after t0, thinned to roughly one point every g frames.
            production = one_series[t0:]
            production_steps = x_axis[t0:]
            picked = timeseries.subsample_correlated_data(production, g=g)
            independent = production[picked]
            independent_steps = production_steps[picked]

            mean = independent.mean()
            discarded = 100 * t0 / len(one_series)

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot()

            # Thermodynamic quantities have no segments, so no spread band.
            if error is not None:
                ax.fill_between(x_axis, one_series - error, one_series + error,
                                color='0.8', linewidth=0, label='within-frame spread')
            ax.plot(x_axis, one_series, color='0.45', linewidth=0.9, label=column)
            ax.axvspan(x_axis[t0], x_axis[-1], color='seagreen', alpha=0.10)
            ax.axvline(x_axis[t0], color='crimson', linewidth=1.5,
                       label=f't0 = {x_axis[t0]:,} ({discarded:.1f}% discarded)')
            ax.plot(independent_steps, independent, 'o', color='seagreen',
                    markersize=3, label=f'{len(independent)} independent samples')
            ax.axhline(mean, color='navy', linestyle='--', linewidth=1.0,
                       label=f'production mean = {mean:.4g}')

            ax.set_xlabel('Timestep')
            ax.set_ylabel(column)
            ax.set_title(f'{bead_type} - {column}   (g = {g:.2f}, Neff = {neff:.1f})')
            ax.legend(fontsize=8, loc='best')

            # A cut past halfway means the series never settled, so say so on the
            # figure rather than leaving the number to be misread as a late but
            # successful equilibration.
            if discarded > 50:
                ax.text(0.02, 0.95, 'NOT EQUILIBRATED: t0 past halfway',
                        transform=ax.transAxes, fontsize=9, color='crimson',
                        weight='bold', va='top')
            start_offset = equilibration[bead_type][column]['start_offset']
            if start_offset > 5:
                ax.text(0.02, 0.85, f'TRANSIENT NOT CUT: first kept point is '
                                    f'{start_offset:.0f} spreads from the mean',
                        transform=ax.transAxes, fontsize=9, color='crimson',
                        weight='bold', va='top')
            if neff < 20:
                ax.text(0.02, 0.90, f'UNDERSAMPLED: Neff = {neff:.1f}',
                        transform=ax.transAxes, fontsize=9, color='crimson',
                        weight='bold', va='top')

            plt.tight_layout()
            plt.savefig(os.path.join(plot_folder, f'{bead_type}_{column}.png'), dpi=150)
            plt.close(fig)

    print(f"wrote {len(to_plot)} figures to {plot_folder}/")
