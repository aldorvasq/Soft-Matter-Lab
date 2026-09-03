import MDAnalysis as mda
import numpy as np
import soft_matter as sm


# Walk the trajectory once and build the gyration tensor eigenvalues for every
# segment of every frame.
#
# The indices coming out of select_positions.py are static: they are read from the
# topology and do not change frame to frame. So the trajectory is the only thing
# that has to be iterated, and it is iterated exactly once for all bead types.
#
# Shapes:
#   idxs_dic[bead_type]    (n_segments, segment_length)      integers, static
#   positions[idx]         (n_segments, segment_length, 3)   one frame of coordinates
#   eigenvalues[bead]      (n_frames, n_segments, 3)         magnitudes, ascending
#   eigenvectors[bead]     (n_frames, n_segments, 3, 3)      orientation, columns
#
# Both halves of the eigen-decomposition are kept. The eigenvalues alone are
# rotation invariant, so they can describe a segment's shape but cannot say which
# lab axis an extent lies along; recovering the in-plane and interface-normal
# components needs the eigenvectors as well.


def gtensor_timeseries(polymer_system, idxs_dic, box, verbose=True):
    """Gyration tensor eigen-decomposition for each segment, for every frame.

    polymer_system : MDAnalysis.Universe
    idxs_dic       : dict of bead_type -> (n_segments, segment_length) atom indices
    box            : (3,) box lengths, passed straight through to soft_matter

    Returns (eigenvalues, eigenvectors, steps). The first two are dicts keyed the
    same way as idxs_dic, holding (n_frames, n_segments, 3) and
    (n_frames, n_segments, 3, 3) arrays respectively; steps is the MD timestep of
    each frame. Both dicts feed straight into the soft_matter functions, which take
    one segment's (3,) eigenvalues and (3, 3) eigenvectors at a time.
    """
    n_frames = len(polymer_system.trajectory)

    # One output array per bead type, since each has a different segment count.
    eigenvalues = {bead_type: np.zeros((n_frames, idx.shape[0], 3))
                   for bead_type, idx in idxs_dic.items()}
    eigenvectors = {bead_type: np.zeros((n_frames, idx.shape[0], 3, 3))
                    for bead_type, idx in idxs_dic.items()}
    steps = np.zeros(n_frames, dtype=np.int64)

    report_every = max(1, n_frames // 10)

    for ts in polymer_system.trajectory:
        # Coordinates of every atom in this frame, fetched once and then sliced
        # for each bead type rather than re-read.
        frame_positions = polymer_system.atoms.positions

        # The GSD reader exposes the real MD timestep when it can; fall back to the
        # frame number so the array always holds something usable.
        steps[ts.frame] = ts.data.get('step', ts.frame) if hasattr(ts, 'data') else ts.frame

        for bead_type, idx in idxs_dic.items():
            # Fancy indexing with a 2-D index array does the segmentation in one
            # step: (n_segments, segment_length) of indices in, coordinates out
            # with a trailing xyz axis.
            segments = frame_positions[idx]

            # soft_matter.gyration_tensor takes one segment at a time, so this
            # loop is what lets the tested implementation be reused unchanged.
            # It returns (eigenvectors, eigenvalues) in that order.
            for s in range(segments.shape[0]):
                (eigenvectors[bead_type][ts.frame, s, :, :],
                 eigenvalues[bead_type][ts.frame, s, :]) = sm.gyration_tensor(
                    segments[s], box)

        if verbose and (ts.frame + 1) % report_every == 0:
            print(f"  frame {ts.frame + 1}/{n_frames}")

    return eigenvalues, eigenvectors, steps

def e2e_timeseries(polymer_system, idxs_dic, box, verbose=True):
    """End-to-end distance of each segment, for every frame.

    Same arguments as gtensor_timeseries. Returns (end_to_end, steps), where
    end_to_end is a dict keyed like idxs_dic holding (n_frames, n_segments) arrays.
    One scalar per segment per frame, not three, because the end-to-end distance is
    a single distance rather than a tensor.

    Two things to keep in mind about the result:
      - soft_matter.end_to_end_distance uses the minimum image convention, so it
        cannot report a distance longer than half the box. That is fine for the
        short sidechains and marginal for the 100-bead backbones.
      - It takes the first and last bead of the segment as the two ends, so it only
        means something for a linear segment. For a whole bottlebrush those two
        beads are a backbone bead and a sidechain bead, which is not an end-to-end
        distance in any useful sense.
    """
    n_frames = len(polymer_system.trajectory)

    # One scalar per segment per frame, so one axis fewer than the tensor arrays.
    end_to_end = {bead_type: np.zeros((n_frames, idx.shape[0]))
                  for bead_type, idx in idxs_dic.items()}
    steps = np.zeros(n_frames, dtype=np.int64)

    report_every = max(1, n_frames // 10)

    for ts in polymer_system.trajectory:
        # Coordinates of every atom in this frame, fetched once and then sliced
        # for each bead type rather than re-read.
        frame_positions = polymer_system.atoms.positions

        # The GSD reader exposes the real MD timestep when it can; fall back to the
        # frame number so the array always holds something usable.
        steps[ts.frame] = ts.data.get('step', ts.frame) if hasattr(ts, 'data') else ts.frame

        for bead_type, idx in idxs_dic.items():
            # Same fancy indexing as above: (n_segments, segment_length) of indices
            # in, coordinates out with a trailing xyz axis.
            segments = frame_positions[idx]

            # soft_matter.end_to_end_distance takes one segment at a time and reads
            # only its first and last bead.
            for s in range(segments.shape[0]):
                end_to_end[bead_type][ts.frame, s] = sm.end_to_end_distance(
                    segments[s], box)

        if verbose and (ts.frame + 1) % report_every == 0:
            print(f"  frame {ts.frame + 1}/{n_frames}")

    return end_to_end, steps
