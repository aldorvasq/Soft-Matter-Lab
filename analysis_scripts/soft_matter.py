import MDAnalysis as mda
import numpy as np

def gyration_tensor(pos, box):

    # Compute the gyration tensor of a set of positions in a periodic box.
    # pos is an (N, 3) array of positions, box is a (3,) array of box lengths.

    box = np.asarray(box[:3], dtype=float)

    theta = (pos / box) * 2.0 * np.pi
    p1av = np.cos(theta).mean(axis=0)
    p2av = np.sin(theta).mean(axis=0)
    theta_av = np.arctan2(-p2av, -p1av) + np.pi
    com = box * theta_av / (2.0 * np.pi)

    dr = (pos - com) - box * np.round((pos - com) / box)
    gm = (dr.T @ dr) / len(pos)                      # gyration tensor

    # eigh returns eigenvalues ascending, and the eigenvectors as the COLUMNS of v,
    # so v[:, k] is the axis belonging to w[k]. The eigenvectors are what carry the
    # orientation: the eigenvalues alone are rotation invariant and cannot say which
    # lab axis a given extent lies along.
    w, v = np.linalg.eigh(gm)
    return v, w

def radius_of_gyration(gm_eigenvalues):
    # Compute the radius of gyration of a set of positions in a periodic box.
    w = np.asarray(gm_eigenvalues, dtype=float)
    rg2_total = w[0] + w[1] + w[2]
    return np.sqrt(rg2_total)

def radius_of_gyration_components(gm_eigenvalues, gm_eigenvectors):
    # Compute the radius of gyration components in the LAB frame.
    #
    # "In-plane" and "interface normal" are statements about the lab axes, not about
    # the molecule's own principal axes, so they cannot be read off the eigenvalues:
    # those are sorted by magnitude and carry no orientation. The lab-frame variances
    # are the diagonal of the gyration tensor, which is recovered from the
    # eigen-decomposition gm = v diag(w) v.T as
    #
    #     gm[i, i] = sum_k v[i, k]**2 * w[k]
    #
    # i.e. each eigenvalue contributes to axis i in proportion to how much its
    # eigenvector points along i.
    w = np.asarray(gm_eigenvalues, dtype=float)
    v = np.asarray(gm_eigenvectors, dtype=float)

    rg2_x, rg2_y, rg2_z = (v ** 2) @ w       # diagonal of the gyration tensor

    # The trace is rotation invariant, so the total is the same either way.
    rg2_total = w[0] + w[1] + w[2]
    rg2_parallel = rg2_x + rg2_y             # in-plane (xy)
    rg2_perpendicular = rg2_z                # interface normal (z)
    return np.sqrt(rg2_total), np.sqrt(rg2_parallel), np.sqrt(rg2_perpendicular)

def shape_anisotropy(gm_eigenvalues):
    # Compute the shape anisotropy of a set of positions in a periodic box.
    w = np.asarray(gm_eigenvalues, dtype=float)
    A3 = 1 - 3 * (w[2] * w[1] + w[1] * w[0] + w[0] * w[2]) / (w[0] + w[1] + w[2])**2
    return A3

def asphericity(gm_eigenvalues):
    # Compute the asphericity of a set of positions in a periodic box.
    w = np.asarray(gm_eigenvalues, dtype=float)
    asp = w[2] - 0.5 * (w[0] + w[1])
    return asp

def end_to_end_distance(pos, box):
    # Compute the end-to-end distance of a set of positions in a periodic box.
    box = np.asarray(box[:3], dtype=float)
    dr = (pos[-1] - pos[0]) - box * np.round((pos[-1] - pos[0]) / box)
    return np.linalg.norm(dr) # Take the length of the vector using L2 norm, aka, the Euclidean norm. This is equivalent to np.sqrt(np.sum(dr**2)).

def interfacial_tension(pressure_tensor, box, normal_axis=2, n_interfaces=2):
    p = np.asarray(pressure_tensor, dtype=float)
    box = np.asarray(box[:3], dtype=float)

    # Where each diagonal component sits within the 6 stored components.
    diagonal_position = {0: 0, 1: 3, 2: 5}   # xx, yy, zz

    normal_stress = p[..., diagonal_position[normal_axis]]
    tangential_axes = [axis for axis in (0, 1, 2) if axis != normal_axis]
    tangential_stress = [p[..., diagonal_position[axis]] for axis in tangential_axes]

    length_along_normal = box[normal_axis]
    return (length_along_normal / n_interfaces) * (
        normal_stress - 0.5 * (tangential_stress[0] + tangential_stress[1]))
