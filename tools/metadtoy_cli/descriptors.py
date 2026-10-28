"""Descriptor functions shared by batch CLI and a future interactive frontend."""
import itertools
import math
import re


def minimum_image(displacement, cell, pbc):
    """Return the Euclidean nearest image, including skewed/mixed-PBC cells.

    Fractional rounding supplies an initial radius R. Any shorter displacement
    has fractional component j bounded by R * ||inverse_cell[:, j]||. Enumerating
    the integer shifts in these intervals therefore includes the true nearest
    image. Orthogonal cells use their direct componentwise projection.
    """
    import numpy as np
    displacement = np.asarray(displacement, dtype=float)
    if pbc is None:
        raise ValueError("minimum-image distance needs explicit PBC; supply --pbc T/F T/F T/F")
    if not any(pbc):
        return displacement.copy()
    if cell is None:
        raise ValueError("minimum-image distance needs a cell")
    try:
        inverse = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        raise ValueError("singular cell") from None
    fractional = displacement @ inverse
    shifts = np.where(pbc, np.rint(fractional), 0.)
    best = displacement - shifts @ cell
    best_squared = float(best @ best)
    gram = cell @ cell.T
    if np.count_nonzero(gram - np.diag(np.diag(gram))) == 0:
        return best
    radii = math.sqrt(best_squared) * np.linalg.norm(inverse, axis=0)
    ranges = []
    candidates = 1
    for axis in range(3):
        if pbc[axis]:
            # Pad interval endpoints for floating-point boundary roundoff.
            padding = 1e-12 * max(1., abs(fractional[axis]), radii[axis])
            lower = math.ceil(fractional[axis] - radii[axis] - padding)
            upper = math.floor(fractional[axis] + radii[axis] + padding)
            candidates *= upper - lower + 1
            if candidates > 1000000:
                raise ValueError("cell skew requires >1,000,000 periodic images; reduce the cell basis")
            ranges.append(range(lower, upper + 1))
        else:
            ranges.append((0,))
    for candidate in itertools.product(*ranges):
        current = displacement - np.asarray(candidate) @ cell
        squared = float(current @ current)
        if squared < best_squared:
            best, best_squared = current, squared
    return best


class DistanceFeatures:
    """Ordered distance definitions and stable atom selection within one source.

    Config identity='id' refers to the integer tags in the file. identity='row'
    refers to zero-based positions in each frame and requires an explicit
    fixed-order acknowledgement. Type/ID sequence changes are then rejected.
    """
    def __init__(self, config, assume_fixed_order=False):
        allowed = {"schema_version", "identity", "length_unit", "features"}
        if not isinstance(config, dict) or set(config) - allowed or config.get("schema_version") != 1:
            raise ValueError("descriptor config requires schema_version=1 and known keys")
        if config.get("identity") not in ("id", "row"):
            raise ValueError("identity must be id or row")
        if not isinstance(config.get("length_unit"), str) or not config["length_unit"].strip():
            raise ValueError("length_unit must explicitly name the input coordinate unit")
        if config["identity"] == "row" and not assume_fixed_order:
            raise ValueError("row identity requires --assume-fixed-order; same-species swaps cannot be detected")
        features = config.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError("features must be a nonempty ordered list")
        names = {"source", "frame", "step", "time", "split"}
        for feature in features:
            if not isinstance(feature, dict) or set(feature) - {"name", "kind", "atoms", "image"}:
                raise ValueError("unknown descriptor keys")
            name = feature.get("name")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in names:
                raise ValueError("feature names must be unique identifiers, excluding source/frame/step/time/split")
            names.add(name)
            if feature.get("kind") != "distance":
                raise ValueError("first dataset version supports kind=distance")
            atoms = feature.get("atoms")
            if (not isinstance(atoms, list) or len(atoms) != 2 or
                    any(type(atom) is not int or atom < 0 for atom in atoms) or atoms[0] == atoms[1]):
                raise ValueError("distance atoms must be two distinct nonnegative integers")
            if feature.get("image", "minimum") not in ("minimum", "direct"):
                raise ValueError("image must be minimum or direct")
        self.config = config
        self.features = features
        self.names = [feature["name"] for feature in features]
        self.selected = sorted({atom for feature in features for atom in feature["atoms"]})
        self.reference = None

    def evaluate(self, frame):
        import numpy as np
        if self.config["identity"] == "id":
            if frame.atom_ids is None:
                raise ValueError("ID selection requires explicit atom IDs in every frame")
            indices = {atom_id: row for row, atom_id in enumerate(frame.atom_ids)}
            missing = [atom for atom in self.selected if atom not in indices]
            if missing:
                raise ValueError("selected atom IDs missing: %s" % missing)
            signature = tuple(frame.labels[indices[atom]] for atom in self.selected)
        else:
            if max(self.selected) >= len(frame.positions):
                raise ValueError("selected row exceeds atom count")
            indices = {atom: atom for atom in self.selected}
            signature = (len(frame.positions), frame.atom_ids, frame.labels)
        if self.reference is None:
            self.reference = signature
        elif signature != self.reference:
            raise ValueError("atom identity/type sequence changed within source")
        values = []
        for feature in self.features:
            first, second = (indices[atom] for atom in feature["atoms"])
            delta = frame.positions[second] - frame.positions[first]
            if feature.get("image", "minimum") == "minimum":
                delta = minimum_image(delta, frame.cell, frame.pbc)
            distance = float(np.linalg.norm(delta))
            if not math.isfinite(distance) or distance <= 0:
                raise ValueError("distance must be finite and nonzero (gradient undefined at overlap)")
            values.append(distance)
        selection = [{"selector": atom, "row": indices[atom],
                      "id": None if frame.atom_ids is None else frame.atom_ids[indices[atom]],
                      "label": frame.labels[indices[atom]]} for atom in self.selected]
        return values, selection
