"""Reopenable scalar-trajectory reader for FES histograms.

Columns keep their original names. Step tokens remain integers; CV and energy
columns are parsed as finite doubles. Plain text and numeric CSV are supported.
A row represents one equally weighted-in-time observation before bias correction.
"""
import csv
from pathlib import Path
from .hills import finite_number
from .trajectory import integer


class ColvarReader:
    def __init__(self, path, fields=None):
        self.path = Path(path).resolve()
        self.csv = self.path.suffix.lower() == ".csv"
        self.explicit_fields = tuple(fields) if fields else None
        self.fields = self.explicit_fields
        self.header_line = None
        with self.path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                text = line.strip()
                if not text:
                    continue
                header = self._comment_header(text)
                if text.startswith("#") and header is None:
                    continue
                if header is None and not text.startswith("#"):
                    tokens = self._split(text)
                    try:
                        for token in tokens:
                            float(token)
                    except ValueError:
                        header = tuple(tokens)
                    else:
                        if self.fields is None:
                            raise ValueError("COLVAR has no field header; supply --fields in file column order")
                        break
                if header is not None:
                    if self.fields is not None and header != self.fields:
                        raise ValueError("--fields disagrees with the COLVAR header")
                    self.fields, self.header_line = header, line_number
                    break
        if not self.fields or len(set(self.fields)) != len(self.fields):
            raise ValueError("COLVAR requires a nonempty unique field list")
        if any(not name or any(character.isspace() for character in name) for name in self.fields):
            raise ValueError("COLVAR field names cannot contain whitespace")

    def _split(self, text):
        return next(csv.reader([text])) if self.csv else text.split()

    def _comment_header(self, text):
        if text.startswith("#! FIELDS "):
            return tuple(text[len("#! FIELDS "):].split())
        if text.startswith("#"):
            fields = text.lstrip("# ").split()
            if fields and fields[0].lower() in ("step", "time", "timestep"):
                return tuple(fields)
        return None

    def _rows(self):
        """Parse once into scalar lists; records/batches share validation."""
        import math
        import re
        previous_step = previous_time = None
        step_index = self.fields.index("step") if "step" in self.fields else None
        time_index = self.fields.index("time") if "time" in self.fields else None
        step_pattern = re.compile(r"[+-]?\d+")
        index = 0
        with self.path.open(encoding="utf-8-sig") as stream:
            for line_number, line in enumerate(stream, 1):
                text = line.strip()
                if not text:
                    continue
                if text.startswith("#"):
                    header = self._comment_header(text)
                    if header is not None and header != self.fields:
                        raise ValueError("%s:%d: COLVAR field header changed" % (self.path, line_number))
                    continue
                if line_number == self.header_line:
                    if tuple(self._split(text)) != self.fields:
                        raise ValueError("COLVAR field header changed")
                    continue
                try:
                    if not line.endswith("\n"):
                        raise ValueError("incomplete final row; use a completed snapshot")
                    tokens = self._split(text)
                    if len(tokens) != len(self.fields):
                        raise ValueError("row width differs from field header")
                    if step_index is not None and step_pattern.fullmatch(tokens[step_index]) is None:
                        raise ValueError("step must be an integer token")
                    values = [int(token) if column == step_index else float(token)
                              for column, token in enumerate(tokens)]
                    if not all(map(math.isfinite, values)):
                        raise ValueError("COLVAR values must be finite")
                    if step_index is not None:
                        step = values[step_index]
                        if step < 0 or step > 9223372036854775807 or (previous_step is not None and step <= previous_step):
                            raise ValueError("step must be strictly increasing nonnegative int64")
                        previous_step = step
                    if time_index is not None:
                        time = values[time_index]
                        if previous_time is not None and time <= previous_time:
                            raise ValueError("time must be strictly increasing")
                        previous_time = time
                except ValueError as error:
                    raise ValueError("%s:%d: %s" % (self.path, line_number, error)) from None
                yield index, values
                index += 1

    def records(self):
        """Yield named scalar observations while preserving exact integer steps."""
        for index, values in self._rows():
            yield index, dict(zip(self.fields, values))

    def batches(self, batch_size=8192):
        """Yield bounded named arrays; step is int64, other fields are float64.

        The integer column never passes through a floating-point matrix, so
        step values above 2**53 remain exact across batch boundaries.
        """
        import numpy as np
        from itertools import islice
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("sample batch size must be positive")
        iterator = iter(self._rows())
        while True:
            rows = list(islice(iterator, batch_size))
            if not rows:
                return
            indices = np.fromiter((index for index, values in rows), dtype=np.int64, count=len(rows))
            columns = {name: np.fromiter((values[column] for index, values in rows),
                                        dtype=np.int64 if name == "step" else np.float64, count=len(rows))
                       for column, name in enumerate(self.fields)}
            yield indices, columns
