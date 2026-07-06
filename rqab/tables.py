"""w_{c,k}(t) matrix-table and b(c) calibration-table access.

The tables are produced by the C++ binaries ``wck_sweep`` and
``wck_calibrate_b`` (see reproduce.py / README).  ``ensure_w_table`` /
``ensure_b_table`` generate them on demand.
"""
from __future__ import annotations

from bisect import bisect_right
import csv
import math
from pathlib import Path
from typing import Iterable

from .util import BUILD_DIR, RESULTS_DIR, run_checked

SQRT2 = math.sqrt(2.0)


def default_w_table_path(k: int) -> Path:
    return RESULTS_DIR / f"w_table_matrix_k{k}.csv"


def default_b_table_path(k: int) -> Path:
    return RESULTS_DIR / f"b_table_k{k}.csv"


def ensure_w_table(
    w_table_path: Path,
    k: int,
    sweep_bin: Path | None = None,
    auto_generate: bool = True,
) -> None:
    if w_table_path.exists():
        return
    if not auto_generate:
        raise FileNotFoundError(f"w table not found: {w_table_path}")

    sweep = sweep_bin if sweep_bin is not None else (BUILD_DIR / "wck_sweep")
    if not sweep.exists():
        raise FileNotFoundError(
            f"w table is missing ({w_table_path}) and sweep binary not found ({sweep})"
        )
    w_table_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [str(sweep), "--k", str(k), "--out", str(w_table_path)],
        "generate w table",
    )
    print(f"[auto-gen] created w table: {w_table_path}")


def ensure_b_table(
    b_table_path: Path,
    w_table_path: Path,
    k: int,
    calibrate_bin: Path | None = None,
    auto_generate: bool = True,
) -> None:
    if b_table_path.exists():
        return
    if not auto_generate:
        raise FileNotFoundError(f"b table not found: {b_table_path}")

    calibrate = calibrate_bin if calibrate_bin is not None else (BUILD_DIR / "wck_calibrate_b")
    if not calibrate.exists():
        raise FileNotFoundError(
            f"b table is missing ({b_table_path}) and calibration binary not found ({calibrate})"
        )
    b_table_path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            str(calibrate),
            "--k",
            str(k),
            "--w-table",
            str(w_table_path),
            "--out",
            str(b_table_path),
        ],
        "generate b table",
    )
    print(f"[auto-gen] created b table: {b_table_path}")


def pchip_slopes(x: list[float], y: list[float]) -> list[float]:
    n = len(x)
    if n != len(y):
        raise ValueError("pchip_slopes: x/y size mismatch")
    if n < 2:
        raise ValueError("pchip_slopes: at least two points required")
    if n == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return [slope, slope]

    h = [x[i + 1] - x[i] for i in range(n - 1)]
    for hi in h:
        if hi <= 0.0:
            raise ValueError("pchip_slopes: x must be strictly increasing")
    delta = [(y[i + 1] - y[i]) / h[i] for i in range(n - 1)]

    m = [0.0] * n
    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0:
            m[k] = 0.0
        elif (delta[k - 1] > 0.0) != (delta[k] > 0.0):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    m0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if (m0 > 0.0) != (delta[0] > 0.0):
        m0 = 0.0
    elif ((delta[0] > 0.0) != (delta[1] > 0.0)) and abs(m0) > abs(3.0 * delta[0]):
        m0 = 3.0 * delta[0]
    m[0] = m0

    mn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if (mn > 0.0) != (delta[-1] > 0.0):
        mn = 0.0
    elif ((delta[-1] > 0.0) != (delta[-2] > 0.0)) and abs(mn) > abs(3.0 * delta[-1]):
        mn = 3.0 * delta[-1]
    m[-1] = mn

    return m


def pchip_eval_scalar(x: list[float], y: list[float], m: list[float], xq: float) -> float:
    n = len(x)
    if n < 2:
        return y[0]
    if xq <= x[0]:
        return y[0]
    if xq >= x[-1]:
        return y[-1]

    i = bisect_right(x, xq) - 1
    if i >= n - 1:
        i = n - 2
    h = x[i + 1] - x[i]
    s = (xq - x[i]) / h

    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2

    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1] + h11 * h * m[i + 1]


def make_log_grid(t_min: float, t_max: float, n: int) -> list[float]:
    if n < 2:
        raise ValueError("log grid needs n >= 2")
    if not (t_min > 0.0 and t_max > t_min):
        raise ValueError("require 0 < t_min < t_max")
    lo = math.log(t_min)
    hi = math.log(t_max)
    step = (hi - lo) / float(n - 1)
    return [math.exp(lo + step * i) for i in range(n)]


class WTableInterpolator:
    """Interpolates w_{c,k}(t) from a wck_sweep matrix CSV.

    PCHIP in log(t) within each c-row; linear blend across c; exponential
    tails outside the c-grid; w(c, 0) = 1 exactly.
    """

    def __init__(self, c_grid: list[float], t_grid: list[float], w_matrix: list[list[float]]) -> None:
        if not c_grid:
            raise ValueError("c-grid is empty")
        if not t_grid:
            raise ValueError("t-grid is empty")
        if len(c_grid) != len(w_matrix):
            raise ValueError("c-grid and matrix row count mismatch")
        if any(len(row) != len(t_grid) for row in w_matrix):
            raise ValueError("inconsistent matrix row size")
        if any(c_grid[i + 1] <= c_grid[i] for i in range(len(c_grid) - 1)):
            raise ValueError("c-grid must be strictly increasing")
        if any(t_grid[i + 1] <= t_grid[i] for i in range(len(t_grid) - 1)):
            raise ValueError("t-grid must be strictly increasing")

        self.c_grid = c_grid
        self.t_grid = t_grid
        self.w_matrix = [row[:] for row in w_matrix]

        if self.t_grid[0] > 0.0:
            self.t_grid.insert(0, 0.0)
            for row in self.w_matrix:
                row.insert(0, 1.0)
        elif abs(self.t_grid[0]) < 1e-14:
            self.t_grid[0] = 0.0
            for row in self.w_matrix:
                row[0] = 1.0
        else:
            raise ValueError("t-grid must start at t>=0")

        self._enforce_properties()
        self._build_row_models()

        self.c_min = self.c_grid[0]
        self.c_max = self.c_grid[-1]
        self.t_min_pos = self._t_positive[0]
        self.t_max = self._t_positive[-1]
        self._c_tail_scale = max(1.0, 0.2 * (self.c_max - self.c_min))

    @classmethod
    def from_matrix_csv(cls, path: Path) -> "WTableInterpolator":
        if not path.exists():
            raise FileNotFoundError(f"table not found: {path}")
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            rows = [row for row in reader if row]
        if len(rows) < 2:
            raise ValueError("table must have header and at least one data row")

        header = rows[0]
        if not header or header[0].strip().lower() != "c":
            raise ValueError("matrix header must start with 'c'")
        t_grid = [float(x) for x in header[1:]]
        if not t_grid:
            raise ValueError("table has no t columns")

        c_vals: list[float] = []
        matrix: list[list[float]] = []
        for row in rows[1:]:
            if len(row) != len(header):
                raise ValueError("inconsistent row length in matrix table")
            c_vals.append(float(row[0]))
            matrix.append([float(x) for x in row[1:]])

        paired = sorted(zip(c_vals, matrix), key=lambda z: z[0])
        c_sorted = [z[0] for z in paired]
        m_sorted = [z[1] for z in paired]
        return cls(c_sorted, t_grid, m_sorted)

    def _enforce_properties(self) -> None:
        for row in self.w_matrix:
            row[0] = 1.0

    def _build_row_models(self) -> None:
        pos_idx = [i for i, t in enumerate(self.t_grid) if t > 0.0]
        if not pos_idx:
            raise ValueError("table has no positive t grid points")
        self._pos_idx = pos_idx
        self._t_positive = [self.t_grid[i] for i in pos_idx]
        self._x_log_positive = [math.log(t) for t in self._t_positive]
        self._row_models: list[tuple[list[float], list[float], list[float]]] = []

        for row in self.w_matrix:
            y = [row[i] for i in pos_idx]
            if len(y) == 1:
                slopes = [0.0]
            else:
                slopes = pchip_slopes(self._x_log_positive, y)
            self._row_models.append((self._t_positive, y, slopes))

    def _interp_row_t(self, row_index: int, t: float) -> float:
        if t <= 0.0:
            return 1.0

        t_pos, y_pos, slopes = self._row_models[row_index]
        t0 = t_pos[0]
        if t <= t0:
            w0 = y_pos[0]
            return 1.0 - (1.0 - w0) * (t / t0)

        if t >= t_pos[-1]:
            return y_pos[-1]

        xq = math.log(t)
        return pchip_eval_scalar(self._x_log_positive, y_pos, slopes, xq)

    def w(self, c: float, t: float) -> float:
        if t <= 0.0:
            return 1.0

        if c <= self.c_min:
            w_edge = self._interp_row_t(0, t)
            val = 1.0 - (1.0 - w_edge) * math.exp((c - self.c_min) / self._c_tail_scale)
            return val

        if c >= self.c_max:
            w_edge = self._interp_row_t(len(self.c_grid) - 1, t)
            val = w_edge * math.exp(-(c - self.c_max) / self._c_tail_scale)
            return val

        i = bisect_right(self.c_grid, c) - 1
        if i >= len(self.c_grid) - 1:
            i = len(self.c_grid) - 2
        c0 = self.c_grid[i]
        c1 = self.c_grid[i + 1]
        w0 = self._interp_row_t(i, t)
        w1 = self._interp_row_t(i + 1, t)
        theta = (c - c0) / (c1 - c0)
        return (1.0 - theta) * w0 + theta * w1

    def curve(self, c: float, t_values: Iterable[float]) -> list[float]:
        return [self.w(c, t) for t in t_values]


class BCalibrationInterpolator:
    """Linear interpolation of b(c) from a wck_calibrate_b CSV.

    Below the c-grid b -> sqrt(2) (the underloaded limit); above it the last
    table value is held constant.
    """

    def __init__(self, c_values: list[float], b_values: list[float]) -> None:
        if not c_values or not b_values:
            raise ValueError("b table is empty")
        if len(c_values) != len(b_values):
            raise ValueError("b table c/b size mismatch")
        for i in range(len(c_values) - 1):
            if not (c_values[i + 1] > c_values[i]):
                raise ValueError("b table c-grid must be strictly increasing")
        self.c_values = c_values
        self.b_values = b_values

    @classmethod
    def from_csv(cls, path: Path) -> "BCalibrationInterpolator":
        if not path.exists():
            raise FileNotFoundError(f"b table not found: {path}")

        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"b table is empty: {path}")
            required = {"c", "b"}
            if not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"b table must contain columns c,b: {path}")

            rows: list[tuple[float, float]] = []
            for row in reader:
                rows.append((float(row["c"]), float(row["b"])))
        if not rows:
            raise ValueError(f"b table has no rows: {path}")
        rows.sort(key=lambda x: x[0])
        return cls([r[0] for r in rows], [r[1] for r in rows])

    def evaluate(self, c: float) -> float:
        if c < self.c_values[0]:
            return SQRT2
        if c > self.c_values[-1]:
            return self.b_values[-1]
        if c == self.c_values[0]:
            return self.b_values[0]

        lo = 0
        hi = len(self.c_values) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self.c_values[mid] <= c:
                lo = mid
            else:
                hi = mid
        c0 = self.c_values[lo]
        c1 = self.c_values[hi]
        b0 = self.b_values[lo]
        b1 = self.b_values[hi]

        if abs(c - c0) <= 1e-14:
            return b0
        if abs(c1 - c) <= 1e-14:
            return b1
        theta = (c - c0) / (c1 - c0)
        return (1.0 - theta) * b0 + theta * b1
