# -*- coding: utf-8 -*-

# Quantum_Jacobians_Hessians_Newton

"""
QUANTUM JACOBIANS HESSIANS NEWTON V2 — Loto 7/39

V2:
- aktivni SEED = 39;
- deterministicka inicijalizacija bez slucajnog generisanja;
- analiticki Jakobijan i Hesijan-vektor tokom ucenja;
- stabilniji izvod bez kvadriranja malih Bornovih tezina;
- Newtonov postupak sa oblascu poverenja;
- eksplicitni zavrsni Jakobijani i puni Hesijan;
- kontrolne tacke vezane za kod, konfiguraciju i podatke.

Kvantni deo:
Qiskit kolo proizvodi kompleksne amplitude.

Sedam nezavisno pripremljenih identicnih registara
ima po sest kubita. Zbog istih ulaza i parametara,
njihov zajednicki pojedinacni izlaz simulira se jednom.

Izmedju registara nema kvantnih kapija.

Raspodela kombinacije:
    P(S | istorija) = product_{j in S} q_j / e_7(q)

q_j su Qiskit Bornove tezine.
e_7 je elementarni simetricni polinom sedmog stepena.

Normalizacija, izvodi i Newtonov optimizator su klasicni.
Nema klasicnog prediktivnog regresora.

Nema:
- klasifikatora;
- slucajnog generisanja kombinacija;
- uzorkovanja merenja;
- mesanja istorije;
- kozmetickih filtera.

Trening koristi proizvode Hesijana i vektora.
Puni Hesijan racuna se jednom za zavrsni model svakog CSV-a.
"""


import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import csv
import hashlib
import math
import sys
import time
import traceback

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import qiskit

from qiskit import QuantumCircuit
from qiskit.circuit.library import RXGate, RYGate, RZZGate
from qiskit.quantum_info import Statevector


import random

random.seed(39)
np.random.seed(39)


VERSION = "QUANTUM_JACOBIANS_HESSIANS_NEWTON_V2_20261031"

SEED = 39


FILES = (
    Path("/data/loto7_4712_k87.csv"),
    Path("/data/loto7_4712_k87_loto_2978.csv"),
    Path("/data/loto7_4712_k87_loto_plus_1734.csv"),
)

NUMBERS = 39
PICKS = 7

QUBITS = 6
BASIS_SIZE = 1 << QUBITS
LAYERS = 2

RIDGES = (0.01, 0.10)
VALIDATION_TIE = 1.0e-5

MAX_ITERATIONS = 24
MAX_ORACLE_CALLS = 120
MAX_CG_STEPS = 4

CG_RELATIVE_TOLERANCE = 0.10

INITIAL_RADIUS = 0.5
MAX_RADIUS = 2.0
MIN_RADIUS = 1.0e-7

ACCEPTANCE_RATIO = 0.10

GRADIENT_TOLERANCE = 1.0e-5
CURVATURE_TOLERANCE = 1.0e-6

if "__file__" in globals():
    SCRIPT_PATH = Path(__file__).resolve()
    SCRIPT_DIR = SCRIPT_PATH.parent
    CODE_HASH = hashlib.sha256(
        SCRIPT_PATH.read_bytes()
    ).hexdigest()
else:
    SCRIPT_PATH = None
    SCRIPT_DIR = Path.cwd()
    CODE_HASH = hashlib.sha256(
        VERSION.encode("utf-8")
    ).hexdigest()

CACHE_DIR = (
    SCRIPT_DIR
    / "Quantum_Jacobians_Hessians_Newton_V2_checkpoints"
)

CONFIGURATION = (
    VERSION,
    SEED,
    QUBITS,
    LAYERS,
    RIDGES,
    VALIDATION_TIE,
    MAX_ITERATIONS,
    MAX_ORACLE_CALLS,
    MAX_CG_STEPS,
    CG_RELATIVE_TOLERANCE,
    INITIAL_RADIUS,
    MAX_RADIUS,
    MIN_RADIUS,
    ACCEPTANCE_RATIO,
    GRADIENT_TOLERANCE,
    CURVATURE_TOLERANCE,
    qiskit.__version__,
    np.__version__,
)


def say(*values):
    print(*values, flush=True)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def finite(values, label):
    require(
        np.all(np.isfinite(values)),
        f"{label}: nekonacan rezultat.",
    )


def next_power_of_two(number):
    require(number >= 1, "Broj uzoraka mora biti pozitivan.")
    return 1 << (int(number) - 1).bit_length()


def digest_items(*items):
    digest = hashlib.sha256()

    for item in items:
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)

            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(array.tobytes())
        else:
            digest.update(repr(item).encode("utf-8"))

    return digest.hexdigest()


def atomic_npz(path, **values):
    temporary = path.with_name(
        path.name + f".{os.getpid()}.tmp"
    )

    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


def maximum_run(numbers):
    longest = 1
    current = 1

    for previous, value in zip(numbers[:-1], numbers[1:]):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1

    return longest


def log_merge(left, right, dleft=None, dright=None):
    total = np.logaddexp(left, right)

    if dleft is None:
        return total, None

    valid = np.isfinite(total)

    weight_left = np.zeros_like(total)
    weight_right = np.zeros_like(total)

    weight_left[valid] = np.exp(
        left[valid] - total[valid]
    )

    weight_right[valid] = np.exp(
        right[valid] - total[valid]
    )

    derivative = (
        weight_left * dleft
        + weight_right * dright
    )

    return total, derivative


class FixedSizeDistribution:
    def __init__(self, numbers, picks):
        self.numbers = int(numbers)
        self.picks = int(picks)

        require(
            1 <= self.picks <= self.numbers,
            "Neispravna velicina kombinacije.",
        )

    def statistics(self, log_weights, direction=None):
        x = np.asarray(log_weights, dtype=np.float64)

        require(
            x.ndim == 2
            and x.shape[1] == self.numbers
            and len(x) > 0,
            "Neispravan oblik tezina raspodele.",
        )

        finite(x, "Logaritamske tezine")

        directional = direction is not None

        if directional:
            dx = np.asarray(direction, dtype=np.float64)

            require(
                dx.shape == x.shape,
                "Neispravan smerni izvod tezina.",
            )

            finite(dx, "Smerni izvod tezina")
        else:
            dx = None

        n = len(x)
        m = self.numbers
        k = self.picks

        prefix = np.full(
            (n, m + 1, k + 1),
            -np.inf,
            dtype=np.float64,
        )

        suffix = np.full_like(prefix, -np.inf)

        prefix[:, 0, 0] = 0.0
        suffix[:, m, 0] = 0.0

        dprefix = (
            np.zeros_like(prefix)
            if directional
            else None
        )

        dsuffix = (
            np.zeros_like(suffix)
            if directional
            else None
        )

        for j in range(m):
            prefix[:, j + 1, :] = prefix[:, j, :]

            if directional:
                dprefix[:, j + 1, :] = dprefix[:, j, :]

            left = prefix[:, j, 1:]
            right = x[:, j, None] + prefix[:, j, :k]

            if directional:
                value, derivative = log_merge(
                    left,
                    right,
                    dprefix[:, j, 1:],
                    dx[:, j, None] + dprefix[:, j, :k],
                )

                dprefix[:, j + 1, 1:] = derivative
            else:
                value, _ = log_merge(left, right)

            prefix[:, j + 1, 1:] = value

        for j in range(m - 1, -1, -1):
            suffix[:, j, :] = suffix[:, j + 1, :]

            if directional:
                dsuffix[:, j, :] = dsuffix[:, j + 1, :]

            left = suffix[:, j + 1, 1:]
            right = x[:, j, None] + suffix[:, j + 1, :k]

            if directional:
                value, derivative = log_merge(
                    left,
                    right,
                    dsuffix[:, j + 1, 1:],
                    dx[:, j, None] + dsuffix[:, j + 1, :k],
                )

                dsuffix[:, j, 1:] = derivative
            else:
                value, _ = log_merge(left, right)

            suffix[:, j, 1:] = value

        log_z = prefix[:, m, k]
        finite(log_z, "Normalizacija raspodele")

        require(
            np.max(
                np.abs(log_z - suffix[:, 0, k])
            ) < 2.0e-9,
            "Prednja i zadnja normalizacija se ne slazu.",
        )

        dlog_z = (
            dprefix[:, m, k]
            if directional
            else None
        )

        marginal = np.empty_like(x)

        dmarginal = (
            np.empty_like(x)
            if directional
            else None
        )

        for j in range(m):
            terms = (
                prefix[:, j, :k]
                + suffix[:, j + 1, k - 1::-1]
            )

            leave_one = np.logaddexp.reduce(
                terms, axis=1
            )

            marginal[:, j] = np.exp(
                x[:, j] + leave_one - log_z
            )

            if directional:
                dterms = (
                    dprefix[:, j, :k]
                    + dsuffix[:, j + 1, k - 1::-1]
                )

                term_weights = np.exp(
                    terms - leave_one[:, None]
                )

                dleave_one = np.sum(
                    term_weights * dterms,
                    axis=1,
                )

                dmarginal[:, j] = marginal[:, j] * (
                    dx[:, j]
                    + dleave_one
                    - dlog_z
                )

        finite(marginal, "Marginalne verovatnoce")

        require(
            np.min(marginal) >= 0.0
            and np.max(marginal) <= 1.0 + 2.0e-8,
            "Marginalne verovatnoce su izvan opsega.",
        )

        require(
            np.max(
                np.abs(marginal.sum(axis=1) - k)
            ) < 2.0e-7,
            "Zbir marginalnih verovatnoca nije ispravan.",
        )

        if directional:
            finite(dlog_z, "Smerni izvod normalizacije")
            finite(dmarginal, "Smerni izvod marginala")

        return log_z, marginal, dlog_z, dmarginal

    def log_probabilities(self, log_weights, targets):
        targets = np.asarray(targets, dtype=np.int64)

        require(
            targets.shape
            == (len(log_weights), self.picks)
            and np.all(targets >= 0)
            and np.all(targets < self.numbers)
            and np.all(np.diff(targets, axis=1) > 0),
            "Neispravne ciljne kombinacije.",
        )

        log_z = self.statistics(log_weights)[0]

        result = (
            np.take_along_axis(
                log_weights, targets, axis=1
            ).sum(axis=1)
            - log_z
        )

        finite(result, "Logaritmi verovatnoca")

        require(
            np.max(result) <= 2.0e-8,
            "Verovatnoca kombinacije prelazi jedan.",
        )

        return result

    def maximum(self, log_weights):
        values = np.asarray(
            log_weights, dtype=np.float64
        )

        require(
            values.shape == (self.numbers,),
            "Neispravan oblik za trazenje maksimuma.",
        )

        finite(values, "Tezine za maksimum")

        selected = np.sort(
            np.argsort(
                -values, kind="stable"
            )[:self.picks]
        )

        log_z = self.statistics(
            values[None, :]
        )[0][0]

        log_probability = float(
            values[selected].sum() - log_z
        )

        require(
            -math.log(
                math.comb(self.numbers, self.picks)
            ) - 2.0e-8
            <= log_probability
            <= 2.0e-8,
            "Neispravna verovatnoca maksimuma.",
        )

        return selected, log_probability

    def independent_log_z(self, log_weights):
        values = np.asarray(
            log_weights, dtype=np.longdouble
        )

        shift = np.max(values)
        weights = np.exp(values - shift)

        coefficients = np.zeros(
            self.picks + 1,
            dtype=np.longdouble,
        )

        coefficients[0] = 1.0

        for weight in weights:
            for degree in range(self.picks, 0, -1):
                coefficients[degree] += (
                    weight * coefficients[degree - 1]
                )

        require(
            coefficients[self.picks] > 0.0,
            "Nezavisna normalizacija nije pozitivna.",
        )

        return float(
            np.log(coefficients[self.picks])
            + self.picks * shift
        )


DISTRIBUTION = FixedSizeDistribution(NUMBERS, PICKS)


class QuantumBatch:
    def __init__(self, inputs):
        self.inputs = np.asarray(
            inputs, dtype=np.complex128
        )

        require(
            self.inputs.ndim == 2
            and len(self.inputs) > 0,
            "Neispravan paket amplituda.",
        )

        self.samples, self.dimension = self.inputs.shape

        require(
            self.dimension & (self.dimension - 1) == 0,
            "Dimenzija nije stepen dvojke.",
        )

        finite(self.inputs, "Ulazne amplitude")

        require(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(self.inputs) ** 2,
                        axis=1,
                    ) - 1.0
                )
            ) < 2.0e-10,
            "Ulazne amplitude nisu normalizovane.",
        )

        self.padded = next_power_of_two(
            self.samples
        )

        matrix = np.zeros(
            (self.padded, self.dimension),
            dtype=np.complex128,
        )

        matrix[:self.samples] = (
            self.inputs / math.sqrt(self.samples)
        )

        self.packed = matrix.reshape(-1)


def quantum_gate(name, angle):
    if name == "rx":
        return RXGate(float(angle))

    if name == "ry":
        return RYGate(float(angle))

    if name == "rzz":
        return RZZGate(float(angle))

    raise RuntimeError("Nepoznata kapija.")


class QuantumSource:
    def __init__(self, qubits=QUBITS, layers=LAYERS):
        self.qubits = int(qubits)
        self.layers = int(layers)

        require(
            self.qubits >= 2 and self.layers >= 1,
            "Neispravna konfiguracija kvantnog izvora.",
        )

        self.dimension = 1 << self.qubits
        self.size = 3 * self.qubits * self.layers

        self.indices = np.arange(
            self.dimension, dtype=np.int64
        )

    def initial_parameters(self, seed):
        """
        Seme aktivno bira deterministicke pocetne uglove.
        Nema poziva generatoru slucajnih brojeva.
        """
        require(
            isinstance(seed, (int, np.integer)),
            "Seme mora biti ceo broj.",
        )

        positions = (
            np.arange(self.size, dtype=np.float64)
            + 1.0
        )

        return (
            0.08 * np.sin(
                positions * math.sqrt(2.0)
                + 0.01 * int(seed)
            )
            + 0.03 * np.cos(
                positions * math.sqrt(3.0)
                + 0.017 * int(seed)
            )
        )

    def operations(self, parameters):
        values = np.asarray(parameters).reshape(
            self.layers, 3, self.qubits
        )

        result = []

        for layer in range(self.layers):
            base = layer * 3 * self.qubits

            for q in range(self.qubits):
                result.append(
                    (
                        "rx",
                        float(values[layer, 0, q]),
                        (q,),
                        base + q,
                    )
                )

            for q in range(self.qubits):
                result.append(
                    (
                        "rzz",
                        float(values[layer, 1, q]),
                        (q, (q + 1) % self.qubits),
                        base + self.qubits + q,
                    )
                )

            for q in range(self.qubits):
                result.append(
                    (
                        "ry",
                        float(values[layer, 2, q]),
                        (q,),
                        base + 2 * self.qubits + q,
                    )
                )

        return result

    def circuit(self, parameters):
        circuit = QuantumCircuit(self.qubits)

        for name, angle, qubits, _ in self.operations(parameters):
            circuit.append(
                quantum_gate(name, angle),
                list(qubits),
            )

        return circuit

    def generator(self, vector, name, qubits):
        matrix = np.asarray(vector).reshape(
            -1, self.dimension
        )

        q = qubits[0]

        if name == "rx":
            return matrix[
                :, self.indices ^ (1 << q)
            ].reshape(-1)

        if name == "ry":
            bit = (
                (self.indices >> q) & 1
            ).astype(np.float64)

            signs = 1j * (2.0 * bit - 1.0)

            return (
                matrix[:, self.indices ^ (1 << q)]
                * signs[None, :]
            ).reshape(-1)

        if name == "rzz":
            r = qubits[1]

            parity = (
                ((self.indices >> q) & 1)
                ^ ((self.indices >> r) & 1)
            )

            return (
                matrix
                * (1.0 - 2.0 * parity)[None, :]
            ).reshape(-1)

        raise RuntimeError("Nepoznat generator.")

    def forward(self, parameters, batch, direction=None):
        parameters = np.asarray(
            parameters, dtype=np.float64
        )

        require(
            parameters.shape == (self.size,)
            and batch.dimension == self.dimension,
            "Neispravni parametri ili ulaz.",
        )

        finite(parameters, "Kvantni parametri")

        if direction is None:
            state = Statevector(
                batch.packed
            ).evolve(
                self.circuit(parameters)
            )

            tangent = None

        else:
            direction = np.asarray(
                direction, dtype=np.float64
            )

            require(
                direction.shape == parameters.shape,
                "Neispravan smer parametara.",
            )

            finite(direction, "Smer parametara")

            state = Statevector(batch.packed)

            tangent = Statevector(
                np.zeros_like(batch.packed)
            )

            for name, angle, qubits, index in self.operations(
                parameters
            ):
                gate = quantum_gate(name, angle)

                state = state.evolve(
                    gate, qargs=list(qubits)
                )

                tangent = tangent.evolve(
                    gate, qargs=list(qubits)
                )

                tangent = Statevector(
                    tangent.data
                    - 0.5j
                    * direction[index]
                    * self.generator(
                        state.data, name, qubits
                    )
                )

        packed = np.asarray(
            state.data
        ).reshape(
            batch.padded, self.dimension
        )

        amplitudes = (
            packed[:batch.samples]
            * math.sqrt(batch.samples)
        )

        finite(amplitudes, "Qiskit amplitude")

        require(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(amplitudes) ** 2,
                        axis=1,
                    ) - 1.0
                )
            ) < 2.0e-9,
            "Qiskit evolucija nije sacuvala normu.",
        )

        if batch.padded > batch.samples:
            require(
                np.max(
                    np.abs(packed[batch.samples:])
                ) < 2.0e-12,
                "Promenjene su prazne indeksne grane.",
            )

        return (
            np.asarray(state.data),
            (
                None
                if tangent is None
                else np.asarray(tangent.data)
            ),
        )

    def backward(
        self,
        parameters,
        state_data,
        tangent_data,
        adjoint_data,
        dadjoint_data=None,
        direction=None,
    ):
        state = Statevector(state_data)
        adjoint = Statevector(adjoint_data)

        directional = direction is not None

        if directional:
            tangent = Statevector(tangent_data)
            dadjoint = Statevector(dadjoint_data)
        else:
            tangent = None
            dadjoint = None

        gradient = np.zeros(
            self.size, dtype=np.float64
        )

        hvp = (
            np.zeros(self.size, dtype=np.float64)
            if directional
            else None
        )

        for name, angle, qubits, index in reversed(
            self.operations(parameters)
        ):
            generated_state = self.generator(
                state.data, name, qubits
            )

            gradient[index] = float(
                np.real(
                    -1j
                    * np.vdot(
                        adjoint.data, generated_state
                    )
                )
            )

            inverse = quantum_gate(name, -angle)

            if directional:
                generated_tangent = self.generator(
                    tangent.data, name, qubits
                )

                hvp[index] = float(
                    np.real(
                        -1j
                        * (
                            np.vdot(
                                dadjoint.data,
                                generated_state,
                            )
                            + np.vdot(
                                adjoint.data,
                                generated_tangent,
                            )
                        )
                    )
                )

                tangent_before = (
                    tangent.data
                    + 0.5j
                    * direction[index]
                    * generated_state
                )

                dadjoint_before = (
                    dadjoint.data
                    + 0.5j
                    * direction[index]
                    * self.generator(
                        adjoint.data, name, qubits
                    )
                )

                tangent = Statevector(
                    tangent_before
                ).evolve(
                    inverse, qargs=list(qubits)
                )

                dadjoint = Statevector(
                    dadjoint_before
                ).evolve(
                    inverse, qargs=list(qubits)
                )

            state = state.evolve(
                inverse, qargs=list(qubits)
            )

            adjoint = adjoint.evolve(
                inverse, qargs=list(qubits)
            )

        finite(gradient, "Gradijent")

        if directional:
            finite(hvp, "Hesijan puta vektor")

        return gradient, hvp


class Objective:
    def __init__(
        self,
        source,
        distribution,
        inputs,
        targets,
        ridge,
    ):
        self.source = source
        self.distribution = distribution
        self.batch = QuantumBatch(inputs)

        self.targets = np.asarray(
            targets, dtype=np.int64
        )

        self.ridge = float(ridge)

        require(
            self.ridge >= 0.0,
            "Regularizacija mora biti nenegativna.",
        )

        require(
            self.targets.shape
            == (
                self.batch.samples,
                self.distribution.picks,
            )
            and np.all(self.targets >= 0)
            and np.all(
                self.targets < self.distribution.numbers
            )
            and np.all(
                np.diff(self.targets, axis=1) > 0
            ),
            "Neispravne ciljne kombinacije.",
        )

    def evaluate(self, parameters, direction=None):
        parameters = np.asarray(
            parameters, dtype=np.float64
        )

        state, tangent = self.source.forward(
            parameters,
            self.batch,
            direction=direction,
        )

        n = self.batch.samples
        m = self.distribution.numbers
        dimension = self.source.dimension

        packed = state.reshape(
            self.batch.padded, dimension
        )

        amplitudes = (
            packed[:n, :m] * math.sqrt(n)
        )

        q = np.abs(amplitudes) ** 2

        finite(q, "Bornove tezine")

        require(
            np.all(q > 0.0),
            "Kandidat ima nultu Bornovu tezinu.",
        )

        log_q = np.log(q)
        directional = direction is not None

        if directional:
            dpacked = tangent.reshape(
                self.batch.padded, dimension
            )

            damplitudes = (
                dpacked[:n, :m] * math.sqrt(n)
            )

            dq = 2.0 * np.real(
                np.conj(amplitudes) * damplitudes
            )

            dlog_q = dq / q
            finite(dlog_q, "Jakobijan puta smer")
        else:
            dpacked = None
            dq = None
            dlog_q = None

        log_z, marginal, _, dmarginal = (
            self.distribution.statistics(
                log_q,
                direction=dlog_q,
            )
        )

        target_log_weight = np.take_along_axis(
            log_q, self.targets, axis=1
        ).sum(axis=1)

        log_probability = (
            target_log_weight - log_z
        )

        finite(log_probability, "Ciljna verovatnoca")

        require(
            np.max(log_probability) <= 2.0e-8,
            "Ciljna verovatnoca prelazi jedan.",
        )

        nll = -float(np.mean(log_probability))

        residual = marginal.copy()

        residual[
            np.arange(n)[:, None],
            self.targets,
        ] -= 1.0

        coefficient = residual / q
        finite(coefficient, "Adjungovani koeficijent")

        adjoint = np.zeros(
            (self.batch.padded, dimension),
            dtype=np.complex128,
        )

        adjoint[:n, :m] = (
            coefficient * packed[:n, :m]
        )

        if directional:
            # V2:
            # (dmarginal - residual * (dq/q)) / q
            # umesto dmarginal/q - residual*dq/(q*q).
            #
            # Izbegava kvadriranje veoma malog q.
            dcoefficient = (
                dmarginal - residual * dlog_q
            ) / q

            finite(
                dcoefficient,
                "Smerni adjungovani koeficijent",
            )

            dadjoint = np.zeros_like(adjoint)

            dadjoint[:n, :m] = (
                dcoefficient * packed[:n, :m]
                + coefficient * dpacked[:n, :m]
            )

            dadjoint_data = dadjoint.reshape(-1)
        else:
            dadjoint_data = None

        gradient, hvp = self.source.backward(
            parameters,
            state,
            tangent,
            adjoint.reshape(-1),
            dadjoint_data=dadjoint_data,
            direction=direction,
        )

        regularization = (
            0.5
            * self.ridge
            * float(np.mean(parameters ** 2))
        )

        gradient += (
            self.ridge
            * parameters
            / self.source.size
        )

        if directional:
            hvp += (
                self.ridge
                * np.asarray(direction)
                / self.source.size
            )

        loss = nll + regularization

        require(
            math.isfinite(loss),
            "Funkcija cilja nije konacna.",
        )

        finite(gradient, "Ukupan gradijent")

        if directional:
            finite(hvp, "Ukupan Hesijan puta vektor")

        return loss, gradient, hvp


def boundary_step(position, direction, radius):
    a = float(np.dot(direction, direction))
    b = 2.0 * float(np.dot(position, direction))
    c = float(
        np.dot(position, position) - radius * radius
    )

    require(
        a > 0.0,
        "Nulti pravac na granici oblasti poverenja.",
    )

    discriminant = max(
        0.0, b * b - 4.0 * a * c
    )

    tau = (
        -b + math.sqrt(discriminant)
    ) / (2.0 * a)

    require(
        tau >= -1.0e-10,
        "Neispravan presek oblasti poverenja.",
    )

    return max(0.0, tau)


def truncated_newton_step(
    gradient,
    radius,
    hessian_product,
    max_steps=MAX_CG_STEPS,
    relative_tolerance=CG_RELATIVE_TOLERANCE,
):
    require(
        radius > 0.0
        and max_steps >= 1
        and 0.0 < relative_tolerance < 1.0,
        "Neispravna konfiguracija Newtonovog potproblema.",
    )

    position = np.zeros_like(gradient)
    h_position = np.zeros_like(gradient)

    residual = gradient.copy()
    direction = -residual

    squared_residual = float(
        np.dot(residual, residual)
    )

    initial_norm = math.sqrt(squared_residual)

    require(
        initial_norm > 0.0,
        "Newtonov korak je pozvan sa nultim gradijentom.",
    )

    for _ in range(max_steps):
        h_direction = hessian_product(direction)
        finite(h_direction, "Hesijan pravca")

        curvature = float(
            np.dot(direction, h_direction)
        )

        squared_direction = float(
            np.dot(direction, direction)
        )

        if curvature <= 1.0e-12 * squared_direction:
            tau = boundary_step(
                position, direction, radius
            )

            return (
                position + tau * direction,
                h_position + tau * h_direction,
                (
                    "Negativna zakrivljenost."
                    if curvature < 0.0
                    else "Ravna ili vrlo slaba zakrivljenost."
                ),
            )

        alpha = squared_residual / curvature
        candidate = position + alpha * direction

        if np.linalg.norm(candidate) >= radius:
            tau = boundary_step(
                position, direction, radius
            )

            return (
                position + tau * direction,
                h_position + tau * h_direction,
                "Granica oblasti poverenja.",
            )

        position = candidate
        h_position += alpha * h_direction

        new_residual = (
            residual + alpha * h_direction
        )

        new_squared_residual = float(
            np.dot(new_residual, new_residual)
        )

        if (
            math.sqrt(new_squared_residual)
            <= relative_tolerance * initial_norm
        ):
            return (
                position,
                h_position,
                "Dovoljno mali Krylov rezidual.",
            )

        beta = (
            new_squared_residual / squared_residual
        )

        direction = -new_residual + beta * direction
        residual = new_residual
        squared_residual = new_squared_residual

    return (
        position,
        h_position,
        "Dostignut budzet Krylov koraka.",
    )


class OracleBudgetReached(Exception):
    pass


@dataclass
class Fit:
    parameters: np.ndarray
    loss: float
    gradient: float
    message: str
    calls: int
    hessian_calls: int


def fit_model(
    source,
    inputs,
    targets,
    ridge,
    label,
    warm=None,
):
    objective = Objective(
        source,
        DISTRIBUTION,
        inputs,
        targets,
        ridge,
    )

    initial = (
        source.initial_parameters(SEED)
        if warm is None
        else warm.parameters.copy()
    )

    key = digest_items(
        CODE_HASH,
        CONFIGURATION,
        label,
        ridge,
        inputs,
        targets,
        initial,
    )

    path = CACHE_DIR / ("fit_" + key + ".npz")

    parameters = initial.copy()
    best_parameters = initial.copy()

    radius = INITIAL_RADIUS
    iterations = 0
    calls = 0
    hessian_calls = 0
    done = False
    message = "Pocetak."

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    str(saved["version"].item()) == VERSION
                    and int(saved["seed"].item()) == SEED
                    and str(saved["code_hash"].item()) == CODE_HASH,
                    "Kontrolna tacka nema odgovarajuci kod ili seme.",
                )

                parameters = np.asarray(
                    saved["parameters"], dtype=np.float64
                ).copy()

                best_parameters = np.asarray(
                    saved["best_parameters"], dtype=np.float64
                ).copy()

                radius = float(saved["radius"].item())
                iterations = int(saved["iterations"].item())
                calls = int(saved["calls"].item())

                hessian_calls = int(
                    saved["hessian_calls"].item()
                )

                done = bool(saved["done"].item())
                message = str(saved["message"].item())

            require(
                parameters.shape == (source.size,)
                and best_parameters.shape == (source.size,),
                "Neispravan oblik sacuvanih parametara.",
            )

            finite(parameters, "Trenutni parametri")
            finite(best_parameters, "Najbolji parametri")

            require(
                0.0 < radius <= MAX_RADIUS
                and 0 <= iterations <= MAX_ITERATIONS
                and 0 <= calls <= MAX_ORACLE_CALLS
                and 0 <= hessian_calls <= calls,
                "Neispravno stanje kontrolne tacke.",
            )

            say(label, "| ucitana V2 kontrolna tacka.")

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            BadZipFile,
        ) as error:
            say(
                label,
                "| kontrolna tacka nije prihvacena:",
                str(error),
            )

            parameters = initial.copy()
            best_parameters = initial.copy()

            radius = INITIAL_RADIUS
            iterations = 0
            calls = 0
            hessian_calls = 0
            done = False
            message = "Ponovljen pocetak."

    current_loss, current_gradient, _ = objective.evaluate(
        parameters
    )

    best_loss, best_gradient, _ = objective.evaluate(
        best_parameters
    )

    if current_loss < best_loss:
        best_parameters = parameters.copy()
        best_loss = float(current_loss)
        best_gradient = current_gradient.copy()

    start = time.perf_counter()

    say()
    say(label)

    say(
        f"SEED: {SEED} "
        f"| uzorci: {len(inputs)} "
        f"| ridge: {ridge:.5f} "
        f"| parametri: {source.size}"
    )

    def save(done_flag, status):
        atomic_npz(
            path,
            version=np.array(VERSION),
            seed=np.array(SEED, dtype=np.int64),
            code_hash=np.array(CODE_HASH),
            parameters=parameters,
            best_parameters=best_parameters,
            radius=np.array(radius),
            iterations=np.array(iterations, dtype=np.int64),
            calls=np.array(calls, dtype=np.int64),
            hessian_calls=np.array(
                hessian_calls, dtype=np.int64
            ),
            done=np.array(bool(done_flag)),
            message=np.array(str(status)),
        )

    def oracle(candidate, direction=None):
        nonlocal calls, hessian_calls

        if calls >= MAX_ORACLE_CALLS:
            raise OracleBudgetReached()

        calls += 1

        if direction is not None:
            hessian_calls += 1

        if calls == 1 or calls % 5 == 0:
            say(
                f"Proracun {calls}/{MAX_ORACLE_CALLS} "
                f"| Hesijan-vektor {hessian_calls} "
                f"| najbolji cilj {best_loss:.9f} "
                f"| proteklo "
                f"{time.perf_counter() - start:.1f} s"
            )

        return objective.evaluate(
            candidate,
            direction=direction,
        )

    try:
        while not done and iterations < MAX_ITERATIONS:
            if (
                np.max(np.abs(current_gradient))
                <= GRADIENT_TOLERANCE
            ):
                message = "Dostignut uslov malog gradijenta."
                break

            if radius < MIN_RADIUS:
                message = "Oblast poverenja je vrlo mala."
                break

            iterations += 1
            old_radius = radius

            def hessian_product(direction):
                return oracle(
                    parameters, direction=direction
                )[2]

            step, h_step, step_reason = (
                truncated_newton_step(
                    current_gradient,
                    radius,
                    hessian_product,
                )
            )

            finite(step, "Newtonov korak")

            require(
                np.linalg.norm(step) <= radius + 2.0e-8,
                "Korak je izvan oblasti poverenja.",
            )

            predicted_reduction = float(
                -np.dot(current_gradient, step)
                - 0.5 * np.dot(step, h_step)
            )

            if (
                not math.isfinite(predicted_reduction)
                or predicted_reduction <= 0.0
            ):
                radius *= 0.25
                message = "Kvadratni model nema pozitivan pad."
                save(False, message)
                continue

            candidate = parameters + step

            try:
                candidate_loss, candidate_gradient, _ = oracle(
                    candidate
                )

            except OracleBudgetReached:
                raise

            except (
                RuntimeError,
                ValueError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as error:
                radius *= 0.25

                message = (
                    "Kandidat odbijen; oblast poverenja smanjena. "
                    + str(error)
                )

                say(message)
                save(False, message)
                continue

            if candidate_loss < best_loss:
                best_parameters = candidate.copy()
                best_loss = float(candidate_loss)
                best_gradient = candidate_gradient.copy()

            actual_reduction = (
                current_loss - candidate_loss
            )

            ratio = (
                actual_reduction / predicted_reduction
            )

            if ratio < 0.25:
                radius *= 0.25

            elif (
                ratio > 0.75
                and np.linalg.norm(step) >= 0.95 * old_radius
            ):
                radius = min(
                    2.0 * radius, MAX_RADIUS
                )

            accepted = (
                ratio > ACCEPTANCE_RATIO
                and candidate_loss < current_loss
            )

            if accepted:
                parameters = candidate
                current_loss = float(candidate_loss)
                current_gradient = candidate_gradient.copy()

            message = step_reason

            say(
                f"Newton {iterations}/{MAX_ITERATIONS} "
                f"| {'prihvacen' if accepted else 'odbijen'} "
                f"| rho {ratio:.4f} "
                f"| radijus {radius:.4e} "
                f"| Max|grad| "
                f"{np.max(np.abs(current_gradient)):.3e}"
            )

            save(False, message)

        else:
            if not done:
                message = "Dostignut budzet Newtonovih iteracija."

    except OracleBudgetReached:
        message = "Dostignut budzet proracuna izvoda."

    except KeyboardInterrupt:
        save(False, "Korisnicki prekid.")
        raise

    except (
        RuntimeError,
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        message = (
            "Zadrzan najbolji numericki vazeci kandidat. "
            + str(error)
        )

    checked_loss, checked_gradient, _ = objective.evaluate(
        best_parameters
    )

    require(
        abs(checked_loss - best_loss)
        < 2.0e-8 * (1.0 + abs(best_loss)),
        "Ponovljeni proracun cilja se ne slaze.",
    )

    best_loss = float(checked_loss)
    best_gradient = checked_gradient.copy()

    save(True, message)

    gradient_norm = float(
        np.max(np.abs(best_gradient))
    )

    say(f"Najbolji cilj: {best_loss:.9f}")
    say(f"Max|gradijent|: {gradient_norm:.3e}")
    say("Status:", message)

    if gradient_norm > GRADIENT_TOLERANCE:
        say(
            "DIJAGNOSTIKA: stacionarnost nije potvrdjena; "
            "zadrzan je najbolji vazeci kandidat."
        )

    return Fit(
        parameters=best_parameters.copy(),
        loss=best_loss,
        gradient=gradient_norm,
        message=message,
        calls=calls,
        hessian_calls=hessian_calls,
    )


def predict_log_weights(source, fit, inputs):
    batch = QuantumBatch(inputs)

    state, _ = source.forward(
        fit.parameters, batch
    )

    amplitudes = (
        state.reshape(
            batch.padded, source.dimension
        )[:batch.samples, :NUMBERS]
        * math.sqrt(batch.samples)
    )

    weights = np.abs(amplitudes) ** 2

    require(
        np.all(weights > 0.0),
        "Predikcija ima nultu Bornovu tezinu.",
    )

    finite(weights, "Prediktivne Bornove tezine")

    return np.log(weights)


def evaluate(source, fit, inputs, targets, with_hits=False):
    log_weights = predict_log_weights(
        source, fit, inputs
    )

    nll = -float(
        np.mean(
            DISTRIBUTION.log_probabilities(
                log_weights, targets
            )
        )
    )

    if not with_hits:
        return nll

    choices = np.argsort(
        -log_weights,
        axis=1,
        kind="stable",
    )[:, :PICKS]

    occupancy = np.zeros(
        (len(targets), NUMBERS),
        dtype=np.int8,
    )

    occupancy[
        np.arange(len(targets))[:, None],
        targets,
    ] = 1

    hits = np.take_along_axis(
        occupancy, choices, axis=1
    ).sum(axis=1)

    histogram = np.bincount(
        hits.astype(np.int64),
        minlength=PICKS + 1,
    )

    return nll, histogram


def encode_history(draws):
    occupancy = np.zeros(
        (len(draws), NUMBERS),
        dtype=np.float64,
    )

    occupancy[
        np.arange(len(draws))[:, None],
        draws,
    ] = 1.0

    indices = np.arange(len(draws))

    current = occupancy[indices]
    previous = occupancy[np.maximum(indices - 1, 0)]
    older = occupancy[np.maximum(indices - 2, 0)]

    strength = (
        1.0 + current
        + 0.5 * previous
        + 0.25 * older
    )

    phase = math.pi * (
        0.37 * current
        - 0.23 * previous
        + 0.11 * older
    )

    inputs = np.zeros(
        (len(draws), BASIS_SIZE),
        dtype=np.complex128,
    )

    inputs[:, :NUMBERS] = (
        np.sqrt(strength) * np.exp(1j * phase)
    )

    inputs /= np.sqrt(
        np.sum(np.abs(inputs) ** 2, axis=1)
    )[:, None]

    finite(inputs, "Istorijske amplitude")
    return inputs


def load_draws(path):
    raw = path.read_bytes()

    fingerprint = hashlib.sha256(
        raw
    ).hexdigest()

    rows = []
    blanks = 0

    for line_number, row in enumerate(
        csv.reader(
            raw.decode("utf-8-sig").splitlines()
        ),
        start=1,
    ):
        if not row or all(
            not value.strip() for value in row
        ):
            blanks += 1
            continue

        require(
            len(row) == PICKS,
            f"{path.name}, red {line_number}: "
            "potrebno je sedam kolona.",
        )

        try:
            values = [
                int(value.strip()) for value in row
            ]

        except ValueError as error:
            raise RuntimeError(
                f"{path.name}, red {line_number}: "
                "vrednosti moraju biti celi brojevi."
            ) from error

        require(
            len(set(values)) == PICKS
            and min(values) >= 1
            and max(values) <= NUMBERS,
            f"{path.name}, red {line_number}: "
            "neispravna kombinacija.",
        )

        rows.append(
            sorted(value - 1 for value in values)
        )

    require(
        len(rows) >= 100,
        f"{path.name}: potrebno je najmanje 100 kola.",
    )

    draws = np.asarray(
        rows, dtype=np.int64
    )

    say()
    say("ULAZ:", path.name)
    say(f"Redova: {len(draws)} | praznih redova: {blanks}")
    say("SHA256:", fingerprint)
    say("Redosled kola ostaje redosled iz CSV-a.")

    return draws, fingerprint


def explicit_jacobians(source, parameters, next_input):
    """
    J_amplitude: kompleksna matrica 64 x broj_parametara.
    J_born: matrica 39 x broj_parametara.
    J_inclusion: matrica 39 x broj_parametara.
    """

    batch = QuantumBatch(next_input)

    require(
        batch.samples == 1,
        "Eksplicitni NEXT Jakobijan ocekuje jedan ulaz.",
    )

    state, _ = source.forward(
        parameters, batch
    )

    amplitude = state[:source.dimension]
    q = np.abs(amplitude[:NUMBERS]) ** 2

    require(
        np.all(q > 0.0),
        "NEXT Jakobijan ima nultu Bornovu tezinu.",
    )

    log_q = np.log(q)[None, :]

    j_amplitude = np.empty(
        (source.dimension, source.size),
        dtype=np.complex128,
    )

    j_born = np.empty(
        (NUMBERS, source.size),
        dtype=np.float64,
    )

    j_inclusion = np.empty_like(j_born)

    for index in range(source.size):
        direction = np.zeros(source.size)
        direction[index] = 1.0

        _, tangent = source.forward(
            parameters,
            batch,
            direction=direction,
        )

        da = tangent[:source.dimension]

        j_amplitude[:, index] = da

        dq = 2.0 * np.real(
            np.conj(amplitude[:NUMBERS])
            * da[:NUMBERS]
        )

        j_born[:, index] = dq

        j_inclusion[:, index] = (
            DISTRIBUTION.statistics(
                log_q,
                direction=(dq / q)[None, :],
            )[3][0]
        )

    finite(j_amplitude, "Eksplicitni Jakobijan amplituda")
    finite(j_born, "Eksplicitni Bornov Jakobijan")
    finite(j_inclusion, "Eksplicitni Jakobijan ukljucivanja")

    norm_derivative = 2.0 * np.real(
        np.conj(amplitude) @ j_amplitude
    )

    require(
        np.max(np.abs(norm_derivative)) < 2.0e-7,
        "Jakobijan ne cuva izvod norme kvantnog stanja.",
    )

    require(
        np.max(
            np.abs(j_inclusion.sum(axis=0))
        ) < 2.0e-7,
        "Jakobijan ukljucivanja ne cuva zbir sedam.",
    )

    return j_amplitude, j_born, j_inclusion


def explicit_hessian(objective, parameters, key):
    """
    Puni Hesijan regularizovane funkcije cilja.

    Racuna se samo za zavrsni model.
    Svaka zavrsena kolona cuva se za nastavak.
    """

    size = objective.source.size
    path = CACHE_DIR / ("hessian_" + key + ".npz")

    matrix = np.zeros(
        (size, size), dtype=np.float64
    )

    completed = 0

    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as saved:
                require(
                    str(saved["version"].item()) == VERSION
                    and int(saved["seed"].item()) == SEED
                    and str(saved["code_hash"].item()) == CODE_HASH,
                    "Neodgovarajuca kontrolna tacka Hesijana.",
                )

                matrix = np.asarray(
                    saved["matrix"], dtype=np.float64
                ).copy()

                completed = int(
                    saved["completed"].item()
                )

            require(
                matrix.shape == (size, size)
                and 0 <= completed <= size,
                "Neispravna matrica Hesijana.",
            )

            finite(
                matrix[:, :completed],
                "Sacuvane kolone Hesijana",
            )

        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            EOFError,
            BadZipFile,
        ) as error:
            say(
                "Kontrolna tacka Hesijana nije prihvacena:",
                str(error),
            )

            matrix = np.zeros(
                (size, size), dtype=np.float64
            )
            completed = 0

    baseline_loss, baseline_gradient, _ = (
        objective.evaluate(parameters)
    )

    say()
    say(
        f"PUNI HESIJAN: zavrseno {completed}/{size} kolona."
    )

    for index in range(completed, size):
        direction = np.zeros(size)
        direction[index] = 1.0

        loss, gradient, column = objective.evaluate(
            parameters,
            direction=direction,
        )

        require(
            abs(loss - baseline_loss)
            < 2.0e-8 * (1.0 + abs(baseline_loss)),
            "Proracun Hesijana promenio je funkciju cilja.",
        )

        require(
            np.max(
                np.abs(gradient - baseline_gradient)
            )
            < 2.0e-7 * (
                1.0 + np.max(np.abs(baseline_gradient))
            ),
            "Proracun Hesijana ne daje isti gradijent.",
        )

        matrix[:, index] = column

        atomic_npz(
            path,
            version=np.array(VERSION),
            seed=np.array(SEED, dtype=np.int64),
            code_hash=np.array(CODE_HASH),
            matrix=matrix,
            completed=np.array(index + 1, dtype=np.int64),
        )

        if (
            (index + 1) % 3 == 0
            or index + 1 == size
        ):
            say(
                f"Hesijan: {index + 1}/{size} kolona."
            )

    finite(matrix, "Puni Hesijan")

    symmetry_error = float(
        np.max(np.abs(matrix - matrix.T))
    )

    require(
        symmetry_error
        < 2.0e-7 * (
            1.0 + np.max(np.abs(matrix))
        ),
        "Puni Hesijan ne prolazi proveru simetrije.",
    )

    symmetric = 0.5 * (
        matrix + matrix.T
    )

    eigenvalues = np.linalg.eigvalsh(
        symmetric
    )

    finite(eigenvalues, "Spektar Hesijana")

    return symmetric, eigenvalues, symmetry_error


def self_checks():
    say()
    say("Pokrecem V2 matematicke i Qiskit samoprovere.")

    distribution = FixedSizeDistribution(6, 3)

    x = np.array(
        [[-0.2, 0.1, -0.7, 0.4, -0.1, 0.2]],
        dtype=np.float64,
    )

    dx = np.array(
        [[0.3, -0.1, 0.2, 0.05, -0.2, 0.1]],
        dtype=np.float64,
    )

    log_z, marginal, dlog_z, dmarginal = (
        distribution.statistics(x, dx)
    )

    subsets = np.array(
        list(combinations(range(6), 3)),
        dtype=np.int64,
    )

    weights = np.exp(
        x[0, subsets].sum(axis=1)
    )

    probabilities = weights / weights.sum()
    exact_marginal = np.zeros(6)

    for subset, probability in zip(
        subsets, probabilities
    ):
        exact_marginal[subset] += probability

    require(
        abs(
            log_z[0] - math.log(float(weights.sum()))
        ) < 2.0e-12,
        "Normalizacija ne prolazi potpuno nabrajanje.",
    )

    require(
        np.max(
            np.abs(marginal[0] - exact_marginal)
        ) < 2.0e-12,
        "Marginalne verovatnoce nisu ispravne.",
    )

    epsilon = 2.0e-6

    plus = distribution.statistics(
        x + epsilon * dx
    )

    minus = distribution.statistics(
        x - epsilon * dx
    )

    require(
        np.max(
            np.abs(
                (plus[0] - minus[0])
                / (2.0 * epsilon)
                - dlog_z
            )
        ) < 2.0e-8,
        "Izvod normalizacije nije ispravan.",
    )

    require(
        np.max(
            np.abs(
                (plus[1] - minus[1])
                / (2.0 * epsilon)
                - dmarginal
            )
        ) < 2.0e-8,
        "Izvod marginala nije ispravan.",
    )

    selected, log_probability = (
        distribution.maximum(x[0])
    )

    require(
        np.array_equal(
            selected,
            subsets[int(np.argmax(weights))],
        ),
        "Maksimum raspodele nije ispravan.",
    )

    require(
        abs(
            math.exp(log_probability)
            - float(np.max(probabilities))
        ) < 2.0e-12,
        "Verovatnoca maksimuma nije ispravna.",
    )

    require(
        abs(
            distribution.independent_log_z(x[0])
            - log_z[0]
        ) < 2.0e-12,
        "Nezavisna normalizacija nije ispravna.",
    )

    for picks in (1, 6):
        boundary = FixedSizeDistribution(6, picks)

        value = boundary.statistics(
            np.zeros((1, 6))
        )[0][0]

        require(
            abs(
                value - math.log(math.comb(6, picks))
            ) < 2.0e-12,
            "Granicni slucaj raspodele nije ispravan.",
        )

    source = QuantumSource(
        qubits=3, layers=2
    )

    parameters = source.initial_parameters(SEED)

    require(
        np.array_equal(
            parameters,
            source.initial_parameters(SEED),
        ),
        "Inicijalizacija sa SEED=39 nije ponovljiva.",
    )

    require(
        np.max(
            np.abs(
                parameters
                - source.initial_parameters(SEED + 1)
            )
        ) > 1.0e-8,
        "Seme ne utice na inicijalizaciju.",
    )

    inputs = np.empty(
        (3, 8), dtype=np.complex128
    )

    for row in range(3):
        for j in range(8):
            inputs[row, j] = (
                1.0
                + 0.12 * math.sin((row + 1) * (j + 1))
            ) * np.exp(0.17j * row * j)

    inputs /= np.sqrt(
        np.sum(np.abs(inputs) ** 2, axis=1)
    )[:, None]

    targets = np.array(
        [[0, 2, 5], [1, 3, 4], [0, 1, 4]],
        dtype=np.int64,
    )

    objective = Objective(
        source,
        distribution,
        inputs,
        targets,
        ridge=0.03,
    )

    vector = np.cos(
        np.arange(source.size) + 0.7
    )

    vector /= np.linalg.norm(vector)

    loss, gradient, hvp = objective.evaluate(
        parameters,
        direction=vector,
    )

    require(
        math.isfinite(loss),
        "Samoprovera cilja nije konacna.",
    )

    gradient_error = 0.0

    for index in range(source.size):
        theta_plus = parameters.copy()
        theta_minus = parameters.copy()

        theta_plus[index] += epsilon
        theta_minus[index] -= epsilon

        f_plus = objective.evaluate(theta_plus)[0]
        f_minus = objective.evaluate(theta_minus)[0]

        numerical = (
            f_plus - f_minus
        ) / (2.0 * epsilon)

        error = abs(
            numerical - gradient[index]
        )

        gradient_error = max(
            gradient_error, error
        )

        require(
            error
            < 2.0e-5 * (
                1.0 + abs(gradient[index])
            ),
            "Gradijent ne prolazi numericku proveru.",
        )

    g_plus = objective.evaluate(
        parameters + epsilon * vector
    )[1]

    g_minus = objective.evaluate(
        parameters - epsilon * vector
    )[1]

    numerical_hvp = (
        g_plus - g_minus
    ) / (2.0 * epsilon)

    hvp_error = float(
        np.max(
            np.abs(numerical_hvp - hvp)
        )
    )

    require(
        hvp_error
        < 3.0e-5 * (
            1.0 + np.max(np.abs(hvp))
        ),
        "Hesijan-vektor ne prolazi numericku proveru.",
    )

    hessian = np.empty(
        (source.size, source.size)
    )

    for index in range(source.size):
        direction = np.zeros(source.size)
        direction[index] = 1.0

        hessian[:, index] = objective.evaluate(
            parameters,
            direction=direction,
        )[2]

    symmetry_error = float(
        np.max(
            np.abs(hessian - hessian.T)
        )
    )

    require(
        symmetry_error < 2.0e-7,
        "Mali Hesijan nije simetrican.",
    )

    batch = QuantumBatch(inputs)

    state, tangent = source.forward(
        parameters,
        batch,
        direction=vector,
    )

    plus_state = source.forward(
        parameters + epsilon * vector,
        batch,
    )[0]

    minus_state = source.forward(
        parameters - epsilon * vector,
        batch,
    )[0]

    tangent_error = float(
        np.max(
            np.abs(
                (plus_state - minus_state)
                / (2.0 * epsilon)
                - tangent
            )
        )
    )

    require(
        tangent_error < 2.0e-7,
        "Jakobijan amplituda ne prolazi proveru.",
    )

    packed = state.reshape(
        batch.padded, source.dimension
    )

    batch_error = 0.0

    for row in range(len(inputs)):
        direct = Statevector(inputs[row]).evolve(
            source.circuit(parameters)
        )

        error = float(
            np.max(
                np.abs(
                    direct.data
                    - packed[row] * math.sqrt(batch.samples)
                )
            )
        )

        batch_error = max(batch_error, error)

        require(
            error < 2.0e-11,
            "Paketna i pojedinacna simulacija se razlikuju.",
        )

    diagonal = np.array(
        [1.0, 2.0, 4.0]
    )

    small_gradient = np.array(
        [0.2, -0.3, 0.1]
    )

    step, h_step, _ = truncated_newton_step(
        small_gradient,
        radius=2.0,
        hessian_product=lambda v: diagonal * v,
        max_steps=3,
        relative_tolerance=1.0e-12,
    )

    require(
        np.max(
            np.abs(
                step + small_gradient / diagonal
            )
        ) < 2.0e-10,
        "Newtonov kvadratni potproblem nije ispravan.",
    )

    require(
        np.max(
            np.abs(
                h_step - diagonal * step
            )
        ) < 2.0e-10,
        "Hesijan koraka nije ispravan.",
    )

    # Negativna zakrivljenost mora dati ogranicen korak.
    negative_diagonal = np.array(
        [-1.0, 2.0, 3.0]
    )

    negative_step, _, _ = truncated_newton_step(
        np.array([1.0, 0.0, 0.0]),
        radius=0.4,
        hessian_product=lambda v: negative_diagonal * v,
    )

    require(
        abs(np.linalg.norm(negative_step) - 0.4)
        < 2.0e-12,
        "Negativna zakrivljenost ne postuje granicu koraka.",
    )

    # Numericka granica stabilnije formule.
    small_q = np.float64(1.0e-200)

    stable_derivative = (
        0.1 - 0.2 * (small_q / small_q)
    ) / small_q

    require(
        np.isfinite(stable_derivative),
        "Stabilnija formula izvoda nije konacna.",
    )

    say("V2 samoprovere su prosle.")
    say(f"Provereno seme: {SEED}")
    say(f"Greska gradijenta: {gradient_error:.3e}")
    say(f"Greska Hesijan-vektor: {hvp_error:.3e}")
    say(f"Greska simetrije Hesijana: {symmetry_error:.3e}")
    say(f"Greska Jakobijan-vektor: {tangent_error:.3e}")
    say(f"Greska paketne simulacije: {batch_error:.3e}")


def resource_report(loaded, source):
    largest = max(
        len(draws)
        for _, draws, _ in loaded
    )

    padded = next_power_of_two(largest)

    state_bytes = (
        padded * BASIS_SIZE * 16
    )

    dp_bytes = (
        largest
        * (NUMBERS + 1)
        * (PICKS + 1)
        * 8
    )

    estimate = (
        384 * 1024**2
        + 12 * state_bytes
        + 10 * dp_bytes
        + 5 * largest * BASIS_SIZE * 16
    )

    circuit = source.circuit(
        source.initial_parameters(SEED)
    )

    say()
    say("PROCENA RESURSA")
    say(f"SEED: {SEED}")
    say(f"Kubita po registru: {source.qubits}")
    say(f"Identicnih nezavisnih registara: {PICKS}")
    say(f"Naucenih uglova: {source.size}")
    say(f"Kapija jednog naucenog kola: {len(circuit.data)}")
    say(f"Dubina jednog naucenog kola: {circuit.depth()}")

    say(
        f"Zavrsni puni Hesijan: "
        f"{source.size} x {source.size}"
    )

    say(
        f"Konzervativna procena memorije: "
        f"{estimate / 1024**3:.2f} GiB"
    )

    say(
        "Broj kapija ne ukljucuje pripremu ulaznog stanja."
    )

    say(
        "Procena memorije nije izmereni vrh procesa."
    )

    require(
        estimate <= 3.0 * 1024**3,
        "Procenjena memorija prelazi tri GiB. "
        "Trening nije zapocet.",
    )


def process_file(
    path,
    draws,
    fingerprint,
    source,
):
    all_inputs = encode_history(draws)

    inputs = all_inputs[:-1]
    targets = draws[1:]
    next_input = all_inputs[-1:]

    count = len(targets)

    cut60 = int(0.60 * count)
    cut70 = int(0.70 * count)
    cut80 = int(0.80 * count)

    folds = (
        (cut60, cut70),
        (cut70, cut80),
    )

    candidates = []

    say()
    say("=" * 70)
    say("OBRADA:", path.name)
    say(f"SEED: {SEED}")
    say(f"Istorijskih prelaza: {count}")

    for ridge in RIDGES:
        warm = None
        weighted_nll = 0.0
        validation_rows = 0

        for fold_index, (train_end, valid_end) in enumerate(folds):
            fit = fit_model(
                source,
                inputs[:train_end],
                targets[:train_end],
                ridge,
                (
                    f"{path.name} | ridge {ridge:.5f} "
                    f"| validacija {fold_index + 1}"
                ),
                warm=warm,
            )

            score = evaluate(
                source,
                fit,
                inputs[train_end:valid_end],
                targets[train_end:valid_end],
            )

            rows = valid_end - train_end

            weighted_nll += score * rows
            validation_rows += rows
            warm = fit

            say(f"Validacioni NLL: {score:.9f}")

        candidates.append(
            {
                "ridge": ridge,
                "validation": (
                    weighted_nll / validation_rows
                ),
                "warm": warm,
            }
        )

    best_validation = min(
        candidate["validation"]
        for candidate in candidates
    )

    equivalent = [
        candidate
        for candidate in candidates
        if candidate["validation"]
        <= best_validation + VALIDATION_TIE
    ]

    selected = max(
        equivalent,
        key=lambda candidate: candidate["ridge"],
    )

    ridge = selected["ridge"]

    say()
    say("IZBOR REGULARIZACIJE")

    for candidate in candidates:
        say(
            f"Ridge {candidate['ridge']:.5f} "
            f"| validacioni NLL "
            f"{candidate['validation']:.9f}"
        )

    say(f"Izabran ridge: {ridge:.5f}")

    say(
        "Pri validacionoj razlici do "
        f"{VALIDATION_TIE:.6f} NLL "
        "prednost ima jaca regularizacija."
    )

    test_fit = fit_model(
        source,
        inputs[:cut80],
        targets[:cut80],
        ridge,
        f"{path.name} | odvojeni test",
        warm=selected["warm"],
    )

    test_nll, histogram = evaluate(
        source,
        test_fit,
        inputs[cut80:],
        targets[cut80:],
        with_hits=True,
    )

    say()
    say(f"Odvojeni test NLL: {test_nll:.9f}")
    say("Test ne menja izabranu regularizaciju.")

    final_fit = fit_model(
        source,
        inputs,
        targets,
        ridge,
        f"{path.name} | zavrsni trening cele istorije",
        warm=test_fit,
    )

    log_weights = predict_log_weights(
        source,
        final_fit,
        next_input,
    )[0]

    log_z, marginal, _, _ = (
        DISTRIBUTION.statistics(
            log_weights[None, :]
        )
    )

    selected_numbers, log_probability = (
        DISTRIBUTION.maximum(log_weights)
    )

    alternative_log_z = (
        DISTRIBUTION.independent_log_z(
            log_weights
        )
    )

    normalization_error = abs(
        alternative_log_z - float(log_z[0])
    )

    require(
        normalization_error < 2.0e-9,
        "Nezavisna NEXT normalizacija nije prosla.",
    )

    normalization_ratio = math.exp(
        alternative_log_z - float(log_z[0])
    )

    direct = Statevector(
        next_input[0]
    ).evolve(
        source.circuit(final_fit.parameters)
    )

    qiskit_error = float(
        np.max(
            np.abs(
                np.abs(direct.data[:NUMBERS]) ** 2
                - np.exp(log_weights)
            )
        )
    )

    require(
        qiskit_error < 2.0e-10,
        "Pojedinacna Qiskit provera NEXT nije prosla.",
    )

    recent_log_weights = predict_log_weights(
        source,
        final_fit,
        all_inputs[-8:],
    )

    recent_marginal = DISTRIBUTION.statistics(
        recent_log_weights
    )[1]

    sensitivity = float(
        np.max(
            np.ptp(recent_marginal, axis=0)
        )
    )

    audit_key = digest_items(
        CODE_HASH,
        CONFIGURATION,
        path.name,
        fingerprint,
        ridge,
        inputs,
        targets,
        final_fit.parameters,
    )

    full_objective = Objective(
        source,
        DISTRIBUTION,
        inputs,
        targets,
        ridge,
    )

    hessian, eigenvalues, hessian_symmetry_error = (
        explicit_hessian(
            full_objective,
            final_fit.parameters,
            audit_key,
        )
    )

    say()
    say("Racunam eksplicitne NEXT Jakobijane.")

    j_amplitude, j_born, j_inclusion = (
        explicit_jacobians(
            source,
            final_fit.parameters,
            next_input,
        )
    )

    diagnostics_path = CACHE_DIR / (
        "diagnostics_" + audit_key + ".npz"
    )

    atomic_npz(
        diagnostics_path,
        version=np.array(VERSION),
        seed=np.array(SEED, dtype=np.int64),
        code_hash=np.array(CODE_HASH),
        data_sha256=np.array(fingerprint),
        parameters=final_fit.parameters,
        ridge=np.array(ridge),
        next_numbers=selected_numbers + 1,
        next_log_probability=np.array(log_probability),
        inclusion=marginal[0],
        jacobian_amplitudes=j_amplitude,
        jacobian_born=j_born,
        jacobian_inclusion=j_inclusion,
        hessian_regularized_nll=hessian,
        hessian_eigenvalues=eigenvalues,
        hessian_symmetry_error=np.array(
            hessian_symmetry_error
        ),
        test_nll=np.array(test_nll),
    )

    minimum_eigenvalue = float(
        eigenvalues[0]
    )

    maximum_eigenvalue = float(
        eigenvalues[-1]
    )

    jacobian_rank = int(
        np.linalg.matrix_rank(j_inclusion)
    )

    numbers = tuple(
        int(value + 1)
        for value in selected_numbers
    )

    warnings = []

    if maximum_run(numbers) >= 5:
        warnings.append(
            "NEXT sadrzi najmanje pet uzastopnih brojeva."
        )

    if numbers[-1] - numbers[0] <= 9:
        warnings.append(
            "Svih sedam brojeva nalazi se u rasponu od deset brojeva."
        )

    if any(
        np.array_equal(
            selected_numbers, historical
        )
        for historical in draws
    ):
        warnings.append(
            "NEXT ponavlja istorijsku kombinaciju."
        )

    if np.array_equal(
        selected_numbers, draws[-1]
    ):
        warnings.append(
            "NEXT je jednak poslednjem istorijskom kolu."
        )

    if sensitivity < 1.0e-8:
        warnings.append(
            "Izlaz slabo reaguje na proverene istorijske kontekste."
        )

    if final_fit.gradient > GRADIENT_TOLERANCE:
        warnings.append(
            "Stacionarnost zavrsnog treninga nije potvrdjena."
        )

    if test_fit.gradient > GRADIENT_TOLERANCE:
        warnings.append(
            "Test model ima nepotvrdjenu stacionarnost."
        )

    if minimum_eigenvalue < -CURVATURE_TOLERANCE:
        warnings.append(
            "Zavrsni Hesijan ima negativnu zakrivljenost; "
            "lokalni minimum nije potvrdjen."
        )

    say()
    say("REZULTAT:", path.name)
    say(f"SEED: {SEED}")
    say("NEXT:", " ".join(map(str, numbers)))
    say(f"Izabran ridge: {ridge:.5f}")

    say(
        f"Validacioni NLL: "
        f"{selected['validation']:.9f}"
    )

    say(f"Odvojeni test NLL: {test_nll:.9f}")

    say(
        f"Modelna verovatnoca NEXT: "
        f"{math.exp(log_probability):.12g}"
    )

    say(
        f"Zbir svih verovatnoca: "
        f"{normalization_ratio:.12f}"
    )

    say(
        f"Greska logaritamske normalizacije: "
        f"{normalization_error:.3e}"
    )

    say(
        f"Zbir verovatnoca ukljucivanja: "
        f"{marginal[0].sum():.12f}"
    )

    say(
        f"Broj kombinacija obuhvacenih raspodelom: "
        f"{math.comb(NUMBERS, PICKS):,}"
    )

    say(
        "Maksimum je tacan za naucenu proizvodnu raspodelu."
    )

    say(
        "Izbor najvecih naucenih Bornovih tezina "
        "ovde sledi iz oblika zajednicke raspodele; "
        "nisu koriscene istorijske ucestalosti."
    )

    say(f"Greska Qiskit provere: {qiskit_error:.3e}")
    say(f"Osetljivost na istoriju: {sensitivity:.9e}")

    say(
        f"Zavrsni Max|gradijent|: "
        f"{final_fit.gradient:.3e}"
    )

    say(
        f"Optimizacioni proracuni: "
        f"{final_fit.calls}"
    )

    say(
        f"Od toga Hesijan-vektor: "
        f"{final_fit.hessian_calls}"
    )

    say("EKSPPLICITNE MATRICE")
    say("Jakobijan amplituda:", j_amplitude.shape)
    say("Jakobijan Bornovih tezina:", j_born.shape)
    say("Jakobijan verovatnoca ukljucivanja:", j_inclusion.shape)
    say("Puni Hesijan regularizovanog NLL-a:", hessian.shape)

    say(
        f"Rang NEXT Jakobijana ukljucivanja: "
        f"{jacobian_rank}"
    )

    say(
        f"Greska simetrije Hesijana: "
        f"{hessian_symmetry_error:.3e}"
    )

    say(
        f"Najmanja sopstvena vrednost Hesijana: "
        f"{minimum_eigenvalue:.9e}"
    )

    say(
        f"Najveca sopstvena vrednost Hesijana: "
        f"{maximum_eigenvalue:.9e}"
    )

    say("Matrice i dijagnostika:", str(diagnostics_path))

    say(
        "Mali gradijent i numericka zakrivljenost "
        "nisu dokaz globalnog optimuma."
    )

    say("Test histogram pogodaka:")

    for hits, frequency in enumerate(histogram):
        say(f"{hits}: {int(frequency)}")

    mean_hits = float(
        np.dot(
            np.arange(PICKS + 1),
            histogram,
        ) / histogram.sum()
    )

    say(
        f"Prosecan broj pogodaka na testu: "
        f"{mean_hits:.6f}"
    )

    say("Verovatnoce ukljucivanja u NEXT raspodeli:")

    for number, probability in enumerate(
        marginal[0], start=1
    ):
        say(f"{number:2d}: {probability:.9f}")

    for warning in warnings:
        say("DIJAGNOSTIKA:", warning)

    if warnings:
        say(
            "Rezultat nije menjan radi uklanjanja "
            "dijagnostickog obrasca."
        )

    return {
        "name": path.name,
        "next": numbers,
        "ridge": ridge,
        "test_nll": test_nll,
        "inclusion": marginal[0].copy(),
        "sensitivity": sensitivity,
        "last": tuple(
            int(value + 1)
            for value in draws[-1]
        ),
        "minimum_hessian_eigenvalue": minimum_eigenvalue,
    }


def main():
    say("QUANTUM JACOBIANS HESSIANS NEWTON V2 — Loto 7/39")
    say("Python:", sys.version.split()[0])
    say("Qiskit:", qiskit.__version__)
    say("NumPy:", np.__version__)
    say(f"SEED = {SEED}")
    say("SHA256 koda:", CODE_HASH)
    say()

    say(
        "SEED=39 aktivno odredjuje deterministicke "
        "pocetne uglove."
    )

    say(
        "Nema generatora slucajnih brojeva "
        "ni uzorkovanja merenja."
    )

    say(
        "Qiskit racuna kompleksne amplitude "
        "tokom treninga i predikcije."
    )

    say(
        "Sedam identicnih nezavisno pripremljenih "
        "registara ima po sest kubita."
    )

    say(
        "Zbog istog ulaza i parametara "
        "pojedinacni izlaz simulira se jednom."
    )

    say(
        "Izmedju registara nema kvantnog sprezanja."
    )

    say(
        "Zajednicka raspodela uslovljena je na "
        "sedam razlicitih brojeva od 1 do 39."
    )

    say(
        "Jakobijan i Hesijan-vektor racunaju se "
        "analitickim diferenciranjem."
    )

    say(
        "Konacne razlike koriste se samo u samoproverama."
    )

    say(
        "Newtonov postupak i njegovi izvodi su "
        "klasicni numericki deo Qiskit modela."
    )

    say(
        "Nema klasicnog prediktivnog regresora."
    )

    say(
        "Validacija: 60%-70% i 70%-80% prelaza."
    )

    say(
        "Odvojeni test: poslednjih 20% prelaza."
    )

    say(
        "Test je jedan korak unapred: prethodno stvarno "
        "kolo koristi se kao dostupan istorijski ulaz."
    )

    say(
        "Zavrsni trening koristi sve istorijske prelaze."
    )

    say(
        f"Budzet po treningu: {MAX_ITERATIONS} Newtonovih "
        f"iteracija i {MAX_ORACLE_CALLS} proracuna izvoda."
    )

    say(
        f"Najvise {MAX_CG_STEPS} Krylov koraka "
        "po Newtonovom potproblemu."
    )

    say(
        "Pocetna i zavrsna provera funkcije cilja "
        "rade se dodatno."
    )

    say(
        "Puni Hesijan racuna se samo za zavrsni model "
        "svakog CSV-a, uz cuvanje svake kolone."
    )

    say(
        "Zavrsni Jakobijani i Hesijan cuvaju se "
        "u lokalnom NPZ fajlu."
    )

    loaded = []

    for path in FILES:
        draws, fingerprint = load_draws(path)

        loaded.append(
            (path, draws, fingerprint)
        )

    source = QuantumSource()

    resource_report(loaded, source)
    self_checks()

    CACHE_DIR.mkdir(
        parents=True, exist_ok=True
    )

    say()
    say("V2 kontrolne tacke:", str(CACHE_DIR))

    results = []

    for path, draws, fingerprint in loaded:
        results.append(
            process_file(
                path,
                draws,
                fingerprint,
                source,
            )
        )

    say()
    say("=" * 70)
    say("ZAVRSNI NEXT REZULTATI")

    for result in results:
        say(
            f"{result['name']} | NEXT: "
            + " ".join(map(str, result["next"]))
            + f" | seed: {SEED}"
            + f" | ridge: {result['ridge']:.5f}"
            + f" | test NLL: {result['test_nll']:.9f}"
        )

    for first, second in combinations(results, 2):
        if first["next"] == second["next"]:
            difference = float(
                np.max(
                    np.abs(
                        first["inclusion"]
                        - second["inclusion"]
                    )
                )
            )

            say()
            say(
                "DIJAGNOSTIKA: isti NEXT za",
                first["name"],
                "i",
                second["name"],
            )

            say(
                f"Najveca razlika izlaznih verovatnoca: "
                f"{difference:.9e}"
            )

            say(
                f"Osetljivosti na istoriju: "
                f"{first['sensitivity']:.9e} i "
                f"{second['sensitivity']:.9e}"
            )

            say(
                "Isto poslednje istorijsko kolo:",
                first["last"] == second["last"],
            )

            say(
                "Rezultat nije menjan radi "
                "vestackog razlikovanja."
            )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        say()
        say("PREKINUTO NA ZAHTEV KORISNIKA.")

        say(
            "Tokom treninga cuvaju se stanje optimizatora "
            "i najbolji kandidat."
        )

        say(
            "Zavrsene kolone punog Hesijana "
            "cuvaju se zasebno."
        )

        sys.exit(130)

    except (
        OSError,
        UnicodeError,
        csv.Error,
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        say()
        say("ZAUSTAVLJENO:", str(error))

        say(
            "Zavrsni skup od tri NEXT rezultata "
            "nije potvrdjen."
        )

        traceback.print_exc()
        sys.exit(1)



"""
QUANTUM JACOBIANS HESSIANS NEWTON V2 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
NumPy: 1.26.4
SEED = 39
SHA256 koda: fef99fbcfdec023aafb1735cc9dcdd65a809a41ac28b1f5b56495d7f817e1bff

SEED=39 aktivno odredjuje deterministicke pocetne uglove.
Nema generatora slucajnih brojeva ni uzorkovanja merenja.
Qiskit racuna kompleksne amplitude tokom treninga i predikcije.
Sedam identicnih nezavisno pripremljenih registara ima po sest kubita.
Zbog istog ulaza i parametara pojedinacni izlaz simulira se jednom.
Izmedju registara nema kvantnog sprezanja.
Zajednicka raspodela uslovljena je na sedam razlicitih brojeva od 1 do 39.
Jakobijan i Hesijan-vektor racunaju se analitickim diferenciranjem.
Konacne razlike koriste se samo u samoproverama.
Newtonov postupak i njegovi izvodi su klasicni numericki deo Qiskit modela.
Nema klasicnog prediktivnog regresora.
Validacija: 60%-70% i 70%-80% prelaza.
Odvojeni test: poslednjih 20% prelaza.
Test je jedan korak unapred: prethodno stvarno kolo koristi se kao dostupan istorijski ulaz.
Zavrsni trening koristi sve istorijske prelaze.
Budzet po treningu: 24 Newtonovih iteracija i 120 proracuna izvoda.
Najvise 4 Krylov koraka po Newtonovom potproblemu.
Pocetna i zavrsna provera funkcije cilja rade se dodatno.
Puni Hesijan racuna se samo za zavrsni model svakog CSV-a, uz cuvanje svake kolone.
Zavrsni Jakobijani i Hesijan cuvaju se u lokalnom NPZ fajlu.

ULAZ: loto7_4712_k87.csv
Redova: 4712 | praznih redova: 0
SHA256: d5504ffa43c8d7ce477ec31e3a0fff18e06ddae6a9fda656fda8813439750808
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4712_k87_loto_2978.csv
Redova: 2978 | praznih redova: 0
SHA256: 0b1abfe53d71a4c52a467be9254d094b3e48113ae875c2a154dc6229e763d1a2
Redosled kola ostaje redosled iz CSV-a.

ULAZ: loto7_4712_k87_loto_plus_1734.csv
Redova: 1734 | praznih redova: 0
SHA256: f2fd75c1e476476a5691c74e942cfee2e4628ee0662c68433f46a7d9efd32b9e
Redosled kola ostaje redosled iz CSV-a.

PROCENA RESURSA
SEED: 39
Kubita po registru: 6
Identicnih nezavisnih registara: 7
Naucenih uglova: 36
Kapija jednog naucenog kola: 36
Dubina jednog naucenog kola: 16
Zavrsni puni Hesijan: 36 x 36
Konzervativna procena memorije: 0.60 GiB
Broj kapija ne ukljucuje pripremu ulaznog stanja.
Procena memorije nije izmereni vrh procesa.

Pokrecem V2 matematicke i Qiskit samoprovere.
V2 samoprovere su prosle.
Provereno seme: 39
Greska gradijenta: 3.076e-10
Greska Hesijan-vektor: 2.296e-10
Greska simetrije Hesijana: 6.661e-16
Greska Jakobijan-vektor: 3.921e-11
Greska paketne simulacije: 2.289e-16

V2 kontrolne tacke: /Quantum_Jacobians_Hessians_Newton/Quantum_Jacobians_Hessians_Newton_V2_checkpoints

======================================================================
OBRADA: loto7_4712_k87.csv
SEED: 39
Istorijskih prelaza: 4711

loto7_4712_k87.csv | ridge 0.01000 | validacija 1
SEED: 39 | uzorci: 2826 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.994217442 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0046 | radijus 5.0000e-01 | Max|grad| 2.558e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.750821036 | proteklo 1.6 s
Newton 2/24 | prihvacen | rho 1.0179 | radijus 5.0000e-01 | Max|grad| 1.110e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.749913787 | proteklo 3.7 s
Newton 3/24 | prihvacen | rho 1.0048 | radijus 5.0000e-01 | Max|grad| 7.887e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.749622673 | proteklo 5.7 s
Newton 4/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 6.913e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.749433342 | proteklo 7.9 s
Newton 5/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 6.404e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749269320 | proteklo 9.9 s
Newton 6/24 | prihvacen | rho 0.9929 | radijus 5.0000e-01 | Max|grad| 7.754e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749097948 | proteklo 11.9 s
Newton 7/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 7.804e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.748898419 | proteklo 14.0 s
Newton 8/24 | prihvacen | rho 0.9855 | radijus 5.0000e-01 | Max|grad| 9.937e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748656215 | proteklo 16.0 s
Newton 9/24 | prihvacen | rho 0.9961 | radijus 5.0000e-01 | Max|grad| 9.637e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.748352427 | proteklo 18.1 s
Newton 10/24 | prihvacen | rho 0.9749 | radijus 5.0000e-01 | Max|grad| 1.141e-02
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.748012025 | proteklo 20.3 s
Newton 11/24 | prihvacen | rho 0.9923 | radijus 5.0000e-01 | Max|grad| 9.179e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747664792 | proteklo 22.5 s
Newton 12/24 | prihvacen | rho 0.9794 | radijus 5.0000e-01 | Max|grad| 9.554e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747382557 | proteklo 24.6 s
Newton 13/24 | prihvacen | rho 0.9972 | radijus 5.0000e-01 | Max|grad| 7.091e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747163217 | proteklo 26.7 s
Newton 14/24 | prihvacen | rho 0.9897 | radijus 5.0000e-01 | Max|grad| 7.719e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746997570 | proteklo 28.8 s
Newton 15/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 6.072e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746862836 | proteklo 30.9 s
Newton 16/24 | prihvacen | rho 0.9949 | radijus 5.0000e-01 | Max|grad| 6.839e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.746749092 | proteklo 33.0 s
Newton 17/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 5.622e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.746648299 | proteklo 35.1 s
Newton 18/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 6.427e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.746555971 | proteklo 37.1 s
Newton 19/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 5.570e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.746467262 | proteklo 39.2 s
Newton 20/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 6.500e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746377925 | proteklo 41.3 s
Newton 21/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 5.978e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746282971 | proteklo 43.3 s
Newton 22/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 7.199e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746176360 | proteklo 45.4 s
Newton 23/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 6.949e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746050881 | proteklo 47.5 s
Newton 24/24 | prihvacen | rho 0.9957 | radijus 5.0000e-01 | Max|grad| 8.414e-03
Najbolji cilj: 16.745900935
Max|gradijent|: 8.414e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.759718450

loto7_4712_k87.csv | ridge 0.01000 | validacija 2
SEED: 39 | uzorci: 3297 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.747884241 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.747884241 | proteklo 1.9 s
Newton 1/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 6.590e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.747507079 | proteklo 4.1 s
Newton 2/24 | prihvacen | rho 0.9935 | radijus 5.0000e-01 | Max|grad| 9.205e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.747329255 | proteklo 6.2 s
Newton 3/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 4.147e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.747193565 | proteklo 8.3 s
Newton 4/24 | prihvacen | rho 0.9879 | radijus 5.0000e-01 | Max|grad| 7.825e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.747082825 | proteklo 10.4 s
Newton 5/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 3.321e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.746988948 | proteklo 12.5 s
Newton 6/24 | prihvacen | rho 0.9884 | radijus 5.0000e-01 | Max|grad| 6.191e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.746915785 | proteklo 14.7 s
Newton 7/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 2.495e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.746858718 | proteklo 16.8 s
Newton 8/24 | prihvacen | rho 0.9924 | radijus 5.0000e-01 | Max|grad| 4.705e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.746816196 | proteklo 19.0 s
Newton 9/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 1.910e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.746783721 | proteklo 21.2 s
Newton 10/24 | prihvacen | rho 0.9960 | radijus 5.0000e-01 | Max|grad| 3.795e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.746758031 | proteklo 23.4 s
Newton 11/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.603e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.746736875 | proteklo 25.7 s
Newton 12/24 | prihvacen | rho 0.9974 | radijus 5.0000e-01 | Max|grad| 3.379e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.746718315 | proteklo 27.9 s
Newton 13/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 1.459e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746701529 | proteklo 30.1 s
Newton 14/24 | prihvacen | rho 0.9972 | radijus 5.0000e-01 | Max|grad| 3.199e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746685828 | proteklo 32.3 s
Newton 15/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 1.385e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.746670861 | proteklo 34.5 s
Newton 16/24 | prihvacen | rho 0.9967 | radijus 5.0000e-01 | Max|grad| 3.120e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.746656439 | proteklo 36.6 s
Newton 17/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 1.343e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.746642327 | proteklo 38.8 s
Newton 18/24 | prihvacen | rho 0.9961 | radijus 5.0000e-01 | Max|grad| 3.098e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.746628485 | proteklo 41.0 s
Newton 19/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 1.322e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746614699 | proteklo 43.2 s
Newton 20/24 | prihvacen | rho 0.9955 | radijus 5.0000e-01 | Max|grad| 3.122e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746600980 | proteklo 45.4 s
Newton 21/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 1.317e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746587111 | proteklo 47.7 s
Newton 22/24 | prihvacen | rho 0.9949 | radijus 5.0000e-01 | Max|grad| 3.187e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746573130 | proteklo 50.0 s
Newton 23/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 1.325e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.746558801 | proteklo 52.2 s
Newton 24/24 | prihvacen | rho 0.9943 | radijus 5.0000e-01 | Max|grad| 3.292e-03
Najbolji cilj: 16.746544184
Max|gradijent|: 3.292e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.755996654

loto7_4712_k87.csv | ridge 0.10000 | validacija 1
SEED: 39 | uzorci: 2826 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.994383036 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0046 | radijus 5.0000e-01 | Max|grad| 2.556e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.750909114 | proteklo 1.6 s
Newton 2/24 | prihvacen | rho 1.0182 | radijus 5.0000e-01 | Max|grad| 1.094e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.750002267 | proteklo 3.7 s
Newton 3/24 | prihvacen | rho 1.0045 | radijus 5.0000e-01 | Max|grad| 7.480e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.749724709 | proteklo 5.8 s
Newton 4/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 6.420e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.749555435 | proteklo 7.9 s
Newton 5/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 5.614e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749419078 | proteklo 10.0 s
Newton 6/24 | prihvacen | rho 0.9947 | radijus 5.0000e-01 | Max|grad| 6.556e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749286718 | proteklo 12.2 s
Newton 7/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 6.359e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.749144344 | proteklo 14.4 s
Newton 8/24 | prihvacen | rho 0.9900 | radijus 5.0000e-01 | Max|grad| 7.878e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748982604 | proteklo 16.4 s
Newton 9/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 7.716e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.748787266 | proteklo 18.5 s
Newton 10/24 | prihvacen | rho 0.9825 | radijus 5.0000e-01 | Max|grad| 9.731e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.748554365 | proteklo 20.6 s
Newton 11/24 | prihvacen | rho 0.9944 | radijus 5.0000e-01 | Max|grad| 8.778e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.748278374 | proteklo 22.7 s
Newton 12/24 | prihvacen | rho 0.9757 | radijus 5.0000e-01 | Max|grad| 9.995e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747998766 | proteklo 24.8 s
Newton 13/24 | prihvacen | rho 0.9934 | radijus 5.0000e-01 | Max|grad| 7.503e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747743077 | proteklo 26.9 s
Newton 14/24 | prihvacen | rho 0.9835 | radijus 5.0000e-01 | Max|grad| 7.750e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747551154 | proteklo 29.1 s
Newton 15/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 5.675e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747408312 | proteklo 31.2 s
Newton 16/24 | prihvacen | rho 0.9935 | radijus 5.0000e-01 | Max|grad| 6.276e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747299207 | proteklo 33.3 s
Newton 17/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 5.030e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747207811 | proteklo 35.4 s
Newton 18/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 5.850e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747125169 | proteklo 37.5 s
Newton 19/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 5.027e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747045178 | proteklo 39.5 s
Newton 20/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 6.086e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746962244 | proteklo 41.6 s
Newton 21/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 5.581e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746870637 | proteklo 43.7 s
Newton 22/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 6.961e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746764173 | proteklo 45.7 s
Newton 23/24 | prihvacen | rho 0.9970 | radijus 5.0000e-01 | Max|grad| 6.559e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746636492 | proteklo 47.8 s
Newton 24/24 | prihvacen | rho 0.9932 | radijus 5.0000e-01 | Max|grad| 8.003e-03
Najbolji cilj: 16.746486625
Max|gradijent|: 8.003e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.759118603

loto7_4712_k87.csv | ridge 0.10000 | validacija 2
SEED: 39 | uzorci: 3297 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.748373027 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.748373027 | proteklo 1.9 s
Newton 1/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 6.068e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.748017975 | proteklo 4.1 s
Newton 2/24 | prihvacen | rho 0.9932 | radijus 5.0000e-01 | Max|grad| 8.389e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.747863698 | proteklo 6.4 s
Newton 3/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 3.410e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.747753806 | proteklo 8.6 s
Newton 4/24 | prihvacen | rho 0.9886 | radijus 5.0000e-01 | Max|grad| 6.428e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.747671832 | proteklo 10.7 s
Newton 5/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 2.452e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.747608757 | proteklo 13.0 s
Newton 6/24 | prihvacen | rho 0.9903 | radijus 5.0000e-01 | Max|grad| 4.642e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747563591 | proteklo 15.2 s
Newton 7/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 1.679e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747531093 | proteklo 17.3 s
Newton 8/24 | prihvacen | rho 0.9941 | radijus 5.0000e-01 | Max|grad| 3.281e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747508296 | proteklo 19.4 s
Newton 9/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.183e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747491905 | proteklo 21.6 s
Newton 10/24 | prihvacen | rho 0.9969 | radijus 5.0000e-01 | Max|grad| 2.483e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747479565 | proteklo 23.8 s
Newton 11/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 9.213e-04
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747469937 | proteklo 25.9 s
Newton 12/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 2.077e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747461907 | proteklo 28.1 s
Newton 13/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 8.064e-04
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747455027 | proteklo 30.2 s
Newton 14/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 1.858e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747448902 | proteklo 32.4 s
Newton 15/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 7.344e-04
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747443361 | proteklo 34.6 s
Newton 16/24 | prihvacen | rho 0.9977 | radijus 5.0000e-01 | Max|grad| 1.728e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747438268 | proteklo 36.8 s
Newton 17/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 6.826e-04
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747433532 | proteklo 38.9 s
Newton 18/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 1.636e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747429093 | proteklo 41.1 s
Newton 19/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 6.440e-04
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747424887 | proteklo 43.3 s
Newton 20/24 | prihvacen | rho 0.9973 | radijus 5.0000e-01 | Max|grad| 1.573e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747420882 | proteklo 45.5 s
Newton 21/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 6.160e-04
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747417025 | proteklo 47.6 s
Newton 22/24 | prihvacen | rho 0.9971 | radijus 5.0000e-01 | Max|grad| 1.535e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747413294 | proteklo 49.8 s
Newton 23/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 5.972e-04
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747409646 | proteklo 52.0 s
Newton 24/24 | prihvacen | rho 0.9969 | radijus 5.0000e-01 | Max|grad| 1.519e-03
Najbolji cilj: 16.747406067
Max|gradijent|: 1.519e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.754527142

IZBOR REGULARIZACIJE
Ridge 0.01000 | validacioni NLL 16.757857552
Ridge 0.10000 | validacioni NLL 16.756822872
Izabran ridge: 0.10000
Pri validacionoj razlici do 0.000010 NLL prednost ima jaca regularizacija.

loto7_4712_k87.csv | odvojeni test
SEED: 39 | uzorci: 3768 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.748386506 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.748386506 | proteklo 2.0 s
Newton 1/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 2.949e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.748142709 | proteklo 4.2 s
Newton 2/24 | prihvacen | rho 1.0019 | radijus 5.0000e-01 | Max|grad| 4.962e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.748073601 | proteklo 6.6 s
Newton 3/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 1.662e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.748039264 | proteklo 8.8 s
Newton 4/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 3.369e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.748017278 | proteklo 11.1 s
Newton 5/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 1.313e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.748002120 | proteklo 13.3 s
Newton 6/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 2.794e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747990016 | proteklo 15.6 s
Newton 7/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 1.167e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747979899 | proteklo 17.8 s
Newton 8/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 2.562e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747970787 | proteklo 20.1 s
Newton 9/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 1.094e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747962377 | proteklo 22.4 s
Newton 10/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 2.468e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747954334 | proteklo 24.6 s
Newton 11/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 1.057e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747946514 | proteklo 26.8 s
Newton 12/24 | prihvacen | rho 0.9981 | radijus 5.0000e-01 | Max|grad| 2.451e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747938761 | proteklo 29.0 s
Newton 13/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 1.042e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747930969 | proteklo 31.3 s
Newton 14/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 2.491e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747923046 | proteklo 33.5 s
Newton 15/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 1.050e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747914886 | proteklo 35.8 s
Newton 16/24 | prihvacen | rho 0.9968 | radijus 5.0000e-01 | Max|grad| 2.579e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747906429 | proteklo 38.0 s
Newton 17/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 1.116e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747897547 | proteklo 40.2 s
Newton 18/24 | prihvacen | rho 0.9961 | radijus 5.0000e-01 | Max|grad| 2.715e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747888195 | proteklo 42.4 s
Newton 19/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 1.194e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747878207 | proteklo 44.6 s
Newton 20/24 | prihvacen | rho 0.9952 | radijus 5.0000e-01 | Max|grad| 2.902e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747867542 | proteklo 46.8 s
Newton 21/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 1.289e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747855974 | proteklo 49.1 s
Newton 22/24 | prihvacen | rho 0.9942 | radijus 5.0000e-01 | Max|grad| 3.149e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747843460 | proteklo 51.3 s
Newton 23/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 1.405e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747829679 | proteklo 53.6 s
Newton 24/24 | prihvacen | rho 0.9929 | radijus 5.0000e-01 | Max|grad| 3.471e-03
Najbolji cilj: 16.747814581
Max|gradijent|: 3.471e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

Odvojeni test NLL: 16.761194127
Test ne menja izabranu regularizaciju.

loto7_4712_k87.csv | zavrsni trening cele istorije
SEED: 39 | uzorci: 4711 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.750635862 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.750635862 | proteklo 3.9 s
Newton 1/24 | prihvacen | rho 1.0145 | radijus 5.0000e-01 | Max|grad| 1.683e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.750242782 | proteklo 8.2 s
Newton 2/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 9.216e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.750102041 | proteklo 12.4 s
Newton 3/24 | prihvacen | rho 1.0025 | radijus 5.0000e-01 | Max|grad| 5.957e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.750029086 | proteklo 16.6 s
Newton 4/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 3.529e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749979097 | proteklo 20.9 s
Newton 5/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 2.971e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749937497 | proteklo 25.1 s
Newton 6/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 3.370e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.749899968 | proteklo 29.4 s
Newton 7/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 2.996e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.749863181 | proteklo 33.6 s
Newton 8/24 | prihvacen | rho 1.0021 | radijus 5.0000e-01 | Max|grad| 3.411e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.749825406 | proteklo 37.8 s
Newton 9/24 | prihvacen | rho 0.9974 | radijus 5.0000e-01 | Max|grad| 3.210e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.749786062 | proteklo 42.1 s
Newton 10/24 | prihvacen | rho 1.0038 | radijus 5.0000e-01 | Max|grad| 3.621e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.749743578 | proteklo 46.4 s
Newton 11/24 | prihvacen | rho 0.9962 | radijus 5.0000e-01 | Max|grad| 3.595e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.749697425 | proteklo 50.7 s
Newton 12/24 | prihvacen | rho 1.0054 | radijus 5.0000e-01 | Max|grad| 3.976e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.749645670 | proteklo 55.0 s
Newton 13/24 | prihvacen | rho 0.9948 | radijus 5.0000e-01 | Max|grad| 4.171e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.749587649 | proteklo 59.2 s
Newton 14/24 | prihvacen | rho 1.0067 | radijus 5.0000e-01 | Max|grad| 4.475e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.749520632 | proteklo 63.6 s
Newton 15/24 | prihvacen | rho 0.9929 | radijus 5.0000e-01 | Max|grad| 4.983e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.749443661 | proteklo 67.8 s
Newton 16/24 | prihvacen | rho 1.0076 | radijus 5.0000e-01 | Max|grad| 5.127e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.749352597 | proteklo 72.1 s
Newton 17/24 | prihvacen | rho 0.9905 | radijus 5.0000e-01 | Max|grad| 6.101e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.749246039 | proteklo 76.4 s
Newton 18/24 | prihvacen | rho 1.0080 | radijus 5.0000e-01 | Max|grad| 5.949e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.749117473 | proteklo 80.6 s
Newton 19/24 | prihvacen | rho 0.9874 | radijus 5.0000e-01 | Max|grad| 7.612e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.748965078 | proteklo 84.8 s
Newton 20/24 | prihvacen | rho 1.0074 | radijus 5.0000e-01 | Max|grad| 6.942e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.748778510 | proteklo 89.0 s
Newton 21/24 | prihvacen | rho 0.9835 | radijus 5.0000e-01 | Max|grad| 9.585e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.748556400 | proteklo 93.3 s
Newton 22/24 | prihvacen | rho 1.0055 | radijus 5.0000e-01 | Max|grad| 8.055e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.748282906 | proteklo 97.5 s
Newton 23/24 | prihvacen | rho 0.9796 | radijus 5.0000e-01 | Max|grad| 1.194e-02
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747960911 | proteklo 101.8 s
Newton 24/24 | prihvacen | rho 1.0018 | radijus 5.0000e-01 | Max|grad| 9.082e-03
Najbolji cilj: 16.747569248
Max|gradijent|: 9.082e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

PUNI HESIJAN: zavrseno 0/36 kolona.
Hesijan: 3/36 kolona.
Hesijan: 6/36 kolona.
Hesijan: 9/36 kolona.
Hesijan: 12/36 kolona.
Hesijan: 15/36 kolona.
Hesijan: 18/36 kolona.
Hesijan: 21/36 kolona.
Hesijan: 24/36 kolona.
Hesijan: 27/36 kolona.
Hesijan: 30/36 kolona.
Hesijan: 33/36 kolona.
Hesijan: 36/36 kolona.

Racunam eksplicitne NEXT Jakobijane.

REZULTAT: loto7_4712_k87.csv
SEED: 39
NEXT: 6 x 14 y 27 z 30
Izabran ridge: 0.10000
Validacioni NLL: 16.756822872
Odvojeni test NLL: 16.761194127
Modelna verovatnoca NEXT: 1.20375354317e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je tacan za naucenu proizvodnu raspodelu.
Izbor najvecih naucenih Bornovih tezina ovde sledi iz oblika zajednicke raspodele; nisu koriscene istorijske ucestalosti.
Greska Qiskit provere: 6.939e-18
Osetljivost na istoriju: 1.758302623e-01
Zavrsni Max|gradijent|: 9.082e-03
Optimizacioni proracuni: 120
Od toga Hesijan-vektor: 96
EKSPPLICITNE MATRICE
Jakobijan amplituda: (64, 36)
Jakobijan Bornovih tezina: (39, 36)
Jakobijan verovatnoca ukljucivanja: (39, 36)
Puni Hesijan regularizovanog NLL-a: (36, 36)
Rang NEXT Jakobijana ukljucivanja: 36
Greska simetrije Hesijana: 7.416e-14
Najmanja sopstvena vrednost Hesijana: -1.594933572e-02
Najveca sopstvena vrednost Hesijana: 8.519446433e+00
Matrice i dijagnostika: /Users/4c/Desktop/GHQ/Astra/Quantum_Jacobians_Hessians_Newton/Quantum_Jacobians_Hessians_Newton_V2_checkpoints/diagnostics_86754326bfaa428e6dfd881af2493fac957613866b66e11f5875b45ec4ff7df1.npz
Mali gradijent i numericka zakrivljenost nisu dokaz globalnog optimuma.
Test histogram pogodaka:
0: 210
1: 391
2: 244
3: 91
4: 7
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.251326
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.191955893
 2: 0.146454037
 3: 0.179023267
 4: 0.145601406
 5: 0.193619174
 6: 0.241419303
 7: 0.197654588
 8: 0.149399556
 9: 0.247397302
10: 0.137035376
11: 0.145429276
12: 0.140469651
13: 0.220292143
14: 0.236834175
15: 0.177025387
16: 0.145159852
17: 0.150295837
18: 0.163269856
19: 0.245665066
20: 0.135174877
21: 0.193302025
22: 0.207820143
23: 0.157388245
24: 0.160845880
25: 0.138131808
26: 0.162659484
27: 0.259246004
28: 0.151730885
29: 0.259191327
30: 0.281376599
31: 0.149833929
32: 0.164845350
33: 0.167324089
34: 0.163017057
35: 0.149388833
36: 0.164036928
37: 0.166615962
38: 0.148277242
39: 0.165792190
DIJAGNOSTIKA: NEXT ponavlja istorijsku kombinaciju.
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
DIJAGNOSTIKA: Zavrsni Hesijan ima negativnu zakrivljenost; lokalni minimum nije potvrdjen.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4712_k87_loto_2978.csv
SEED: 39
Istorijskih prelaza: 2977

loto7_4712_k87_loto_2978.csv | ridge 0.01000 | validacija 1
SEED: 39 | uzorci: 1786 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.987501367 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0028 | radijus 5.0000e-01 | Max|grad| 2.297e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.752400103 | proteklo 0.8 s
Newton 2/24 | prihvacen | rho 1.0490 | radijus 5.0000e-01 | Max|grad| 1.968e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.751256104 | proteklo 2.0 s
Newton 3/24 | prihvacen | rho 1.0017 | radijus 5.0000e-01 | Max|grad| 1.491e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.750759838 | proteklo 3.1 s
Newton 4/24 | prihvacen | rho 0.9825 | radijus 5.0000e-01 | Max|grad| 2.001e-02
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.750237246 | proteklo 4.3 s
Newton 5/24 | prihvacen | rho 1.0016 | radijus 5.0000e-01 | Max|grad| 1.293e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749656598 | proteklo 5.3 s
Newton 6/24 | prihvacen | rho 0.9779 | radijus 5.0000e-01 | Max|grad| 1.399e-02
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749043342 | proteklo 6.5 s
Newton 7/24 | prihvacen | rho 0.9857 | radijus 5.0000e-01 | Max|grad| 1.397e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.748327333 | proteklo 7.5 s
Newton 8/24 | prihvacen | rho 0.9797 | radijus 5.0000e-01 | Max|grad| 1.221e-02
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747676939 | proteklo 8.6 s
Newton 9/24 | prihvacen | rho 0.9879 | radijus 5.0000e-01 | Max|grad| 1.001e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747177560 | proteklo 9.7 s
Newton 10/24 | prihvacen | rho 0.9926 | radijus 5.0000e-01 | Max|grad| 8.189e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.746839340 | proteklo 10.7 s
Newton 11/24 | prihvacen | rho 0.9947 | radijus 5.0000e-01 | Max|grad| 7.492e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.746591107 | proteklo 11.8 s
Newton 12/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 6.559e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.746395565 | proteklo 12.9 s
Newton 13/24 | prihvacen | rho 0.9966 | radijus 5.0000e-01 | Max|grad| 6.541e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.746227440 | proteklo 14.0 s
Newton 14/24 | prihvacen | rho 0.9976 | radijus 5.0000e-01 | Max|grad| 6.185e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746077792 | proteklo 15.0 s
Newton 15/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 6.226e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.745935866 | proteklo 16.1 s
Newton 16/24 | prihvacen | rho 0.9959 | radijus 5.0000e-01 | Max|grad| 6.039e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.745800165 | proteklo 17.2 s
Newton 17/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 6.080e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.745666595 | proteklo 18.2 s
Newton 18/24 | prihvacen | rho 0.9952 | radijus 5.0000e-01 | Max|grad| 5.742e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.745538387 | proteklo 19.3 s
Newton 19/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 5.852e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.745414277 | proteklo 20.4 s
Newton 20/24 | prihvacen | rho 0.9961 | radijus 5.0000e-01 | Max|grad| 5.519e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.745297440 | proteklo 21.5 s
Newton 21/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 5.609e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.745185912 | proteklo 22.6 s
Newton 22/24 | prihvacen | rho 0.9972 | radijus 5.0000e-01 | Max|grad| 5.282e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.745081206 | proteklo 23.6 s
Newton 23/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 5.418e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.744981119 | proteklo 24.7 s
Newton 24/24 | prihvacen | rho 0.9976 | radijus 5.0000e-01 | Max|grad| 5.066e-03
Najbolji cilj: 16.744886661
Max|gradijent|: 5.066e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.754661138

loto7_4712_k87_loto_2978.csv | ridge 0.01000 | validacija 2
SEED: 39 | uzorci: 2083 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.746293447 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.746293447 | proteklo 1.8 s
Newton 1/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 6.804e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.745554014 | proteklo 3.8 s
Newton 2/24 | prihvacen | rho 0.9885 | radijus 5.0000e-01 | Max|grad| 6.006e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.745438263 | proteklo 5.8 s
Newton 3/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 5.792e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.745359986 | proteklo 7.9 s
Newton 4/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 4.487e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.745293090 | proteklo 9.9 s
Newton 5/24 | prihvacen | rho 0.9983 | radijus 5.0000e-01 | Max|grad| 3.904e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.745235722 | proteklo 11.9 s
Newton 6/24 | prihvacen | rho 0.9980 | radijus 5.0000e-01 | Max|grad| 4.043e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.745186115 | proteklo 13.9 s
Newton 7/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 4.004e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.745142221 | proteklo 15.9 s
Newton 8/24 | prihvacen | rho 0.9983 | radijus 5.0000e-01 | Max|grad| 4.056e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.745100679 | proteklo 17.9 s
Newton 9/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 3.791e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.745061104 | proteklo 19.9 s
Newton 10/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 3.898e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.745022974 | proteklo 21.9 s
Newton 11/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 3.593e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.744986368 | proteklo 23.8 s
Newton 12/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 3.725e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.744951036 | proteklo 25.8 s
Newton 13/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 3.400e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.744917053 | proteklo 27.8 s
Newton 14/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 3.553e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.744884210 | proteklo 29.8 s
Newton 15/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 3.214e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.744852553 | proteklo 31.8 s
Newton 16/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 3.391e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.744821900 | proteklo 33.8 s
Newton 17/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 3.042e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.744792265 | proteklo 35.8 s
Newton 18/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 3.245e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.744763494 | proteklo 37.7 s
Newton 19/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 2.888e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.744735576 | proteklo 39.7 s
Newton 20/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 3.121e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.744708388 | proteklo 41.7 s
Newton 21/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 2.820e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.744681897 | proteklo 43.7 s
Newton 22/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 3.020e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.744656020 | proteklo 45.7 s
Newton 23/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 2.807e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.744630706 | proteklo 47.7 s
Newton 24/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 2.940e-03
Najbolji cilj: 16.744605915
Max|gradijent|: 2.940e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.762488534

loto7_4712_k87_loto_2978.csv | ridge 0.10000 | validacija 1
SEED: 39 | uzorci: 1786 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.987666961 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0028 | radijus 5.0000e-01 | Max|grad| 2.295e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.752487651 | proteklo 0.8 s
Newton 2/24 | prihvacen | rho 1.0481 | radijus 5.0000e-01 | Max|grad| 1.926e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.751348686 | proteklo 1.9 s
Newton 3/24 | prihvacen | rho 1.0031 | radijus 5.0000e-01 | Max|grad| 1.308e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.750909780 | proteklo 3.0 s
Newton 4/24 | prihvacen | rho 0.9861 | radijus 5.0000e-01 | Max|grad| 1.687e-02
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.750508652 | proteklo 4.0 s
Newton 5/24 | prihvacen | rho 1.0029 | radijus 5.0000e-01 | Max|grad| 1.078e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.750091218 | proteklo 5.1 s
Newton 6/24 | prihvacen | rho 0.9823 | radijus 5.0000e-01 | Max|grad| 1.204e-02
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749647627 | proteklo 6.1 s
Newton 7/24 | prihvacen | rho 0.9923 | radijus 5.0000e-01 | Max|grad| 1.263e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.749097041 | proteklo 7.2 s
Newton 8/24 | prihvacen | rho 0.9768 | radijus 5.0000e-01 | Max|grad| 1.270e-02
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748514658 | proteklo 8.2 s
Newton 9/24 | prihvacen | rho 0.9856 | radijus 5.0000e-01 | Max|grad| 1.070e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747977876 | proteklo 9.3 s
Newton 10/24 | prihvacen | rho 0.9868 | radijus 5.0000e-01 | Max|grad| 8.783e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747591118 | proteklo 10.3 s
Newton 11/24 | prihvacen | rho 0.9930 | radijus 5.0000e-01 | Max|grad| 7.127e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747328660 | proteklo 11.4 s
Newton 12/24 | prihvacen | rho 0.9968 | radijus 5.0000e-01 | Max|grad| 6.225e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747145249 | proteklo 12.5 s
Newton 13/24 | prihvacen | rho 0.9964 | radijus 5.0000e-01 | Max|grad| 5.758e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747001117 | proteklo 13.5 s
Newton 14/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 5.244e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746881145 | proteklo 14.6 s
Newton 15/24 | prihvacen | rho 0.9980 | radijus 5.0000e-01 | Max|grad| 5.195e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746774482 | proteklo 15.6 s
Newton 16/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 5.022e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.746676526 | proteklo 16.7 s
Newton 17/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 5.059e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.746581587 | proteklo 17.7 s
Newton 18/24 | prihvacen | rho 0.9965 | radijus 5.0000e-01 | Max|grad| 5.028e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.746488674 | proteklo 18.8 s
Newton 19/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 5.055e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.746395345 | proteklo 19.8 s
Newton 20/24 | prihvacen | rho 0.9955 | radijus 5.0000e-01 | Max|grad| 5.001e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746303438 | proteklo 20.9 s
Newton 21/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 4.976e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746212408 | proteklo 21.9 s
Newton 22/24 | prihvacen | rho 0.9957 | radijus 5.0000e-01 | Max|grad| 4.913e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746124872 | proteklo 23.0 s
Newton 23/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 4.803e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746040351 | proteklo 24.0 s
Newton 24/24 | prihvacen | rho 0.9966 | radijus 5.0000e-01 | Max|grad| 4.720e-03
Najbolji cilj: 16.745960700
Max|gradijent|: 4.720e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.753734796

loto7_4712_k87_loto_2978.csv | ridge 0.10000 | validacija 2
SEED: 39 | uzorci: 2083 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.747183244 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.747183244 | proteklo 1.8 s
Newton 1/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 6.317e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.746464194 | proteklo 3.8 s
Newton 2/24 | prihvacen | rho 0.9957 | radijus 5.0000e-01 | Max|grad| 5.222e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.746360051 | proteklo 5.8 s
Newton 3/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 4.226e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.746302431 | proteklo 7.8 s
Newton 4/24 | prihvacen | rho 0.9983 | radijus 5.0000e-01 | Max|grad| 3.236e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.746259206 | proteklo 9.8 s
Newton 5/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 3.461e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.746225257 | proteklo 11.8 s
Newton 6/24 | prihvacen | rho 0.9976 | radijus 5.0000e-01 | Max|grad| 3.319e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.746194872 | proteklo 13.8 s
Newton 7/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 3.223e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.746166954 | proteklo 16.0 s
Newton 8/24 | prihvacen | rho 0.9981 | radijus 5.0000e-01 | Max|grad| 3.164e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.746140771 | proteklo 18.0 s
Newton 9/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 3.040e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.746116218 | proteklo 20.0 s
Newton 10/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 3.008e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.746093082 | proteklo 22.0 s
Newton 11/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 2.869e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.746071310 | proteklo 24.0 s
Newton 12/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 2.852e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.746050747 | proteklo 26.0 s
Newton 13/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 2.702e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746031361 | proteklo 28.0 s
Newton 14/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 2.696e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746013031 | proteklo 30.0 s
Newton 15/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 2.540e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.745995735 | proteklo 32.1 s
Newton 16/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 2.541e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.745979375 | proteklo 34.1 s
Newton 17/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 2.381e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.745963929 | proteklo 36.1 s
Newton 18/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 2.388e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.745949315 | proteklo 38.1 s
Newton 19/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 2.227e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.745935512 | proteklo 40.0 s
Newton 20/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 2.239e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.745922446 | proteklo 42.0 s
Newton 21/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 2.079e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.745910096 | proteklo 44.0 s
Newton 22/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 2.098e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.745898397 | proteklo 46.0 s
Newton 23/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 1.940e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.745887327 | proteklo 47.9 s
Newton 24/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 1.965e-03
Najbolji cilj: 16.745876829
Max|gradijent|: 1.965e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.761152166

IZBOR REGULARIZACIJE
Ridge 0.01000 | validacioni NLL 16.758581414
Ridge 0.10000 | validacioni NLL 16.757449715
Izabran ridge: 0.10000
Pri validacionoj razlici do 0.000010 NLL prednost ima jaca regularizacija.

loto7_4712_k87_loto_2978.csv | odvojeni test
SEED: 39 | uzorci: 2381 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.747917708 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.747917708 | proteklo 1.8 s
Newton 1/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 7.280e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.747471439 | proteklo 3.9 s
Newton 2/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 3.837e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.747376549 | proteklo 5.9 s
Newton 3/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 2.187e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.747335770 | proteklo 7.9 s
Newton 4/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 2.874e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.747312694 | proteklo 10.0 s
Newton 5/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.538e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.747296664 | proteklo 12.1 s
Newton 6/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 2.161e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747284775 | proteklo 14.3 s
Newton 7/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.224e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747275487 | proteklo 16.4 s
Newton 8/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 1.747e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747267920 | proteklo 18.5 s
Newton 9/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.024e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747261557 | proteklo 20.5 s
Newton 10/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.496e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747256046 | proteklo 22.5 s
Newton 11/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 9.286e-04
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747251165 | proteklo 24.6 s
Newton 12/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 1.336e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747246759 | proteklo 26.6 s
Newton 13/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 8.738e-04
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747242718 | proteklo 28.6 s
Newton 14/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 1.228e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747238971 | proteklo 30.7 s
Newton 15/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 8.350e-04
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747235458 | proteklo 32.7 s
Newton 16/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 1.150e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747232145 | proteklo 34.8 s
Newton 17/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 8.047e-04
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747228995 | proteklo 36.8 s
Newton 18/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 1.090e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747225992 | proteklo 38.8 s
Newton 19/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 7.794e-04
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747223110 | proteklo 40.9 s
Newton 20/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 1.044e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747220343 | proteklo 42.9 s
Newton 21/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 7.572e-04
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747217672 | proteklo 44.9 s
Newton 22/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 1.010e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747215093 | proteklo 47.0 s
Newton 23/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 7.372e-04
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747212594 | proteklo 49.0 s
Newton 24/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 9.830e-04
Najbolji cilj: 16.747210173
Max|gradijent|: 9.830e-04
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

Odvojeni test NLL: 16.737695029
Test ne menja izabranu regularizaciju.

loto7_4712_k87_loto_2978.csv | zavrsni trening cele istorije
SEED: 39 | uzorci: 2977 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.745509845 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.745509845 | proteklo 1.9 s
Newton 1/24 | prihvacen | rho 0.9963 | radijus 5.0000e-01 | Max|grad| 6.179e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.745137599 | proteklo 4.0 s
Newton 2/24 | prihvacen | rho 1.0040 | radijus 5.0000e-01 | Max|grad| 4.543e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.745052081 | proteklo 6.1 s
Newton 3/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 2.935e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.745008311 | proteklo 8.2 s
Newton 4/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 3.359e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.744979165 | proteklo 10.3 s
Newton 5/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 2.400e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.744957553 | proteklo 12.4 s
Newton 6/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 2.795e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.744940092 | proteklo 14.5 s
Newton 7/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 2.104e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.744925369 | proteklo 16.7 s
Newton 8/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 2.498e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.744912352 | proteklo 18.8 s
Newton 9/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 1.922e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.744900627 | proteklo 20.9 s
Newton 10/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 2.316e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.744889813 | proteklo 23.0 s
Newton 11/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.789e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.744879780 | proteklo 25.1 s
Newton 12/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 2.182e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.744870361 | proteklo 27.2 s
Newton 13/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.681e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.744861514 | proteklo 29.3 s
Newton 14/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 2.068e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.744853152 | proteklo 31.4 s
Newton 15/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.586e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.744845262 | proteklo 33.5 s
Newton 16/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.964e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.744837788 | proteklo 35.6 s
Newton 17/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.499e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.744830728 | proteklo 37.8 s
Newton 18/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.864e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.744824041 | proteklo 39.9 s
Newton 19/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.415e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.744817727 | proteklo 42.0 s
Newton 20/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.767e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.744811755 | proteklo 44.1 s
Newton 21/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.334e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.744806123 | proteklo 46.2 s
Newton 22/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 1.671e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.744800806 | proteklo 48.3 s
Newton 23/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.255e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.744795799 | proteklo 50.5 s
Newton 24/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 1.577e-03
Najbolji cilj: 16.744791081
Max|gradijent|: 1.577e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

PUNI HESIJAN: zavrseno 0/36 kolona.
Hesijan: 3/36 kolona.
Hesijan: 6/36 kolona.
Hesijan: 9/36 kolona.
Hesijan: 12/36 kolona.
Hesijan: 15/36 kolona.
Hesijan: 18/36 kolona.
Hesijan: 21/36 kolona.
Hesijan: 24/36 kolona.
Hesijan: 27/36 kolona.
Hesijan: 30/36 kolona.
Hesijan: 33/36 kolona.
Hesijan: 36/36 kolona.

Racunam eksplicitne NEXT Jakobijane.

REZULTAT: loto7_4712_k87_loto_2978.csv
SEED: 39
NEXT: 6 x 15 y 29 z 35
Izabran ridge: 0.10000
Validacioni NLL: 16.757449715
Odvojeni test NLL: 16.737695029
Modelna verovatnoca NEXT: 1.81316930435e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 3.553e-15
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je tacan za naucenu proizvodnu raspodelu.
Izbor najvecih naucenih Bornovih tezina ovde sledi iz oblika zajednicke raspodele; nisu koriscene istorijske ucestalosti.
Greska Qiskit provere: 6.939e-18
Osetljivost na istoriju: 1.577963856e-01
Zavrsni Max|gradijent|: 1.577e-03
Optimizacioni proracuni: 120
Od toga Hesijan-vektor: 96
EKSPPLICITNE MATRICE
Jakobijan amplituda: (64, 36)
Jakobijan Bornovih tezina: (39, 36)
Jakobijan verovatnoca ukljucivanja: (39, 36)
Puni Hesijan regularizovanog NLL-a: (36, 36)
Rang NEXT Jakobijana ukljucivanja: 36
Greska simetrije Hesijana: 8.749e-14
Najmanja sopstvena vrednost Hesijana: -2.066153451e-04
Najveca sopstvena vrednost Hesijana: 8.708191008e+00
Matrice i dijagnostika: /Users/4c/Desktop/GHQ/Astra/Quantum_Jacobians_Hessians_Newton/Quantum_Jacobians_Hessians_Newton_V2_checkpoints/diagnostics_097da0b0b397c0cea17748f9f69776900b131059a93d9b85ee09b701f71094f2.npz
Mali gradijent i numericka zakrivljenost nisu dokaz globalnog optimuma.
Test histogram pogodaka:
0: 132
1: 242
2: 167
3: 48
4: 7
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.255034
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.158017637
 2: 0.150329273
 3: 0.150984570
 4: 0.139868797
 5: 0.152640439
 6: 0.289582948
 7: 0.146394972
 8: 0.155753741
 9: 0.282837946
10: 0.164304690
11: 0.157722170
12: 0.144648392
13: 0.161242466
14: 0.218230627
15: 0.243521638
16: 0.141394873
17: 0.159919928
18: 0.150169275
19: 0.211411001
20: 0.155653952
21: 0.155816230
22: 0.258692067
23: 0.160425793
24: 0.129576424
25: 0.171117385
26: 0.158762962
27: 0.211211239
28: 0.141689971
29: 0.276324957
30: 0.189068837
31: 0.144922186
32: 0.248941292
33: 0.147537317
34: 0.170275768
35: 0.255455142
36: 0.149851706
37: 0.158584809
38: 0.180346058
39: 0.156770522
DIJAGNOSTIKA: NEXT ponavlja istorijsku kombinaciju.
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
DIJAGNOSTIKA: Zavrsni Hesijan ima negativnu zakrivljenost; lokalni minimum nije potvrdjen.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4712_k87_loto_plus_1734.csv
SEED: 39
Istorijskih prelaza: 1733

loto7_4712_k87_loto_plus_1734.csv | ridge 0.01000 | validacija 1
SEED: 39 | uzorci: 1039 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.990232136 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 2.339e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.732424498 | proteklo 0.8 s
Newton 2/24 | prihvacen | rho 1.0161 | radijus 5.0000e-01 | Max|grad| 1.091e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.731466436 | proteklo 1.7 s
Newton 3/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 8.206e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731204559 | proteklo 2.7 s
Newton 4/24 | prihvacen | rho 0.9915 | radijus 5.0000e-01 | Max|grad| 8.135e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731006254 | proteklo 3.7 s
Newton 5/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 9.115e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.730783573 | proteklo 4.6 s
Newton 6/24 | prihvacen | rho 0.9855 | radijus 5.0000e-01 | Max|grad| 9.285e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.730534648 | proteklo 5.6 s
Newton 7/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 9.724e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.730258401 | proteklo 6.5 s
Newton 8/24 | prihvacen | rho 0.9854 | radijus 5.0000e-01 | Max|grad| 9.961e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.729970385 | proteklo 7.5 s
Newton 9/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 9.994e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.729665044 | proteklo 8.4 s
Newton 10/24 | prihvacen | rho 0.9868 | radijus 5.0000e-01 | Max|grad| 1.053e-02
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.729353114 | proteklo 9.4 s
Newton 11/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.011e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.729028907 | proteklo 10.4 s
Newton 12/24 | prihvacen | rho 0.9870 | radijus 5.0000e-01 | Max|grad| 1.060e-02
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.728712017 | proteklo 11.3 s
Newton 13/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 9.618e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.728403538 | proteklo 12.3 s
Newton 14/24 | prihvacen | rho 0.9882 | radijus 5.0000e-01 | Max|grad| 9.883e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.728124558 | proteklo 13.3 s
Newton 15/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 8.550e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.727874377 | proteklo 14.3 s
Newton 16/24 | prihvacen | rho 0.9910 | radijus 5.0000e-01 | Max|grad| 8.648e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.727663198 | proteklo 15.3 s
Newton 17/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 7.407e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.727483160 | proteklo 16.3 s
Newton 18/24 | prihvacen | rho 0.9947 | radijus 5.0000e-01 | Max|grad| 7.601e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.727331491 | proteklo 17.4 s
Newton 19/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 6.754e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.727197862 | proteklo 18.4 s
Newton 20/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 7.139e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.727076629 | proteklo 19.3 s
Newton 21/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 6.608e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.726961092 | proteklo 20.3 s
Newton 22/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 7.096e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.726848533 | proteklo 21.3 s
Newton 23/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 6.731e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.726735195 | proteklo 22.2 s
Newton 24/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 7.279e-03
Najbolji cilj: 16.726620187
Max|gradijent|: 7.279e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.767434658

loto7_4712_k87_loto_plus_1734.csv | ridge 0.01000 | validacija 2
SEED: 39 | uzorci: 1213 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.732485292 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.732485292 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 1.0026 | radijus 5.0000e-01 | Max|grad| 1.247e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.731198132 | proteklo 1.9 s
Newton 2/24 | prihvacen | rho 0.9872 | radijus 5.0000e-01 | Max|grad| 1.335e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.730848853 | proteklo 2.9 s
Newton 3/24 | prihvacen | rho 1.0043 | radijus 5.0000e-01 | Max|grad| 7.028e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.730586775 | proteklo 3.9 s
Newton 4/24 | prihvacen | rho 0.9893 | radijus 5.0000e-01 | Max|grad| 1.348e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.730352952 | proteklo 4.9 s
Newton 5/24 | prihvacen | rho 1.0023 | radijus 5.0000e-01 | Max|grad| 5.740e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.730114389 | proteklo 5.8 s
Newton 6/24 | prihvacen | rho 0.9845 | radijus 5.0000e-01 | Max|grad| 1.405e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.729876394 | proteklo 6.8 s
Newton 7/24 | prihvacen | rho 1.0020 | radijus 5.0000e-01 | Max|grad| 5.568e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.729630873 | proteklo 7.8 s
Newton 8/24 | prihvacen | rho 0.9835 | radijus 5.0000e-01 | Max|grad| 1.387e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.729395574 | proteklo 8.9 s
Newton 9/24 | prihvacen | rho 1.0019 | radijus 5.0000e-01 | Max|grad| 5.194e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.729165015 | proteklo 9.9 s
Newton 10/24 | prihvacen | rho 0.9835 | radijus 5.0000e-01 | Max|grad| 1.294e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.728956600 | proteklo 10.9 s
Newton 11/24 | prihvacen | rho 1.0020 | radijus 5.0000e-01 | Max|grad| 4.553e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.728763613 | proteklo 11.9 s
Newton 12/24 | prihvacen | rho 0.9837 | radijus 5.0000e-01 | Max|grad| 1.164e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.728595547 | proteklo 12.9 s
Newton 13/24 | prihvacen | rho 1.0023 | radijus 5.0000e-01 | Max|grad| 3.894e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.728443844 | proteklo 13.8 s
Newton 14/24 | prihvacen | rho 0.9831 | radijus 5.0000e-01 | Max|grad| 1.067e-02
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.728309497 | proteklo 14.9 s
Newton 15/24 | prihvacen | rho 1.0027 | radijus 5.0000e-01 | Max|grad| 3.437e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.728183888 | proteklo 15.8 s
Newton 16/24 | prihvacen | rho 0.9798 | radijus 5.0000e-01 | Max|grad| 1.056e-02
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.728063846 | proteklo 16.8 s
Newton 17/24 | prihvacen | rho 1.0032 | radijus 5.0000e-01 | Max|grad| 3.432e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.727940794 | proteklo 17.8 s
Newton 18/24 | prihvacen | rho 0.9728 | radijus 5.0000e-01 | Max|grad| 1.156e-02
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.727811558 | proteklo 18.8 s
Newton 19/24 | prihvacen | rho 1.0036 | radijus 5.0000e-01 | Max|grad| 3.870e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.727664203 | proteklo 19.7 s
Newton 20/24 | prihvacen | rho 0.9626 | radijus 5.0000e-01 | Max|grad| 1.385e-02
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.727497144 | proteklo 20.8 s
Newton 21/24 | prihvacen | rho 1.0036 | radijus 5.0000e-01 | Max|grad| 4.662e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.727287268 | proteklo 21.7 s
Newton 22/24 | prihvacen | rho 0.9473 | radijus 5.0000e-01 | Max|grad| 1.803e-02
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.727034445 | proteklo 22.7 s
Newton 23/24 | prihvacen | rho 1.0034 | radijus 5.0000e-01 | Max|grad| 6.019e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.726683215 | proteklo 23.7 s
Newton 24/24 | prihvacen | rho 0.9174 | radijus 5.0000e-01 | Max|grad| 2.558e-02
Najbolji cilj: 16.726240915
Max|gradijent|: 2.558e-02
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.735671594

loto7_4712_k87_loto_plus_1734.csv | ridge 0.10000 | validacija 1
SEED: 39 | uzorci: 1039 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.990397730 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 2.337e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.732511036 | proteklo 0.8 s
Newton 2/24 | prihvacen | rho 1.0161 | radijus 5.0000e-01 | Max|grad| 1.062e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.731557019 | proteklo 1.7 s
Newton 3/24 | prihvacen | rho 0.9983 | radijus 5.0000e-01 | Max|grad| 7.703e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731311235 | proteklo 2.7 s
Newton 4/24 | prihvacen | rho 0.9916 | radijus 5.0000e-01 | Max|grad| 7.703e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731131068 | proteklo 3.6 s
Newton 5/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 8.406e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.730932367 | proteklo 4.6 s
Newton 6/24 | prihvacen | rho 0.9849 | radijus 5.0000e-01 | Max|grad| 8.664e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.730712977 | proteklo 5.5 s
Newton 7/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 8.830e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.730472507 | proteklo 6.5 s
Newton 8/24 | prihvacen | rho 0.9850 | radijus 5.0000e-01 | Max|grad| 9.159e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.730224685 | proteklo 7.4 s
Newton 9/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 8.983e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.729963992 | proteklo 8.4 s
Newton 10/24 | prihvacen | rho 0.9866 | radijus 5.0000e-01 | Max|grad| 9.668e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.729697217 | proteklo 9.4 s
Newton 11/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 9.086e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.729418384 | proteklo 10.3 s
Newton 12/24 | prihvacen | rho 0.9862 | radijus 5.0000e-01 | Max|grad| 9.738e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.729144672 | proteklo 11.3 s
Newton 13/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 8.516e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.728878961 | proteklo 12.2 s
Newton 14/24 | prihvacen | rho 0.9873 | radijus 5.0000e-01 | Max|grad| 8.920e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.728641874 | proteklo 13.2 s
Newton 15/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 7.335e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.728433207 | proteklo 14.1 s
Newton 16/24 | prihvacen | rho 0.9906 | radijus 5.0000e-01 | Max|grad| 7.600e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.728261068 | proteklo 15.1 s
Newton 17/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 6.135e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.728117949 | proteklo 16.0 s
Newton 18/24 | prihvacen | rho 0.9945 | radijus 5.0000e-01 | Max|grad| 6.452e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.728001040 | proteklo 17.0 s
Newton 19/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 5.349e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.727901986 | proteklo 17.9 s
Newton 20/24 | prihvacen | rho 0.9972 | radijus 5.0000e-01 | Max|grad| 5.798e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.727816082 | proteklo 18.9 s
Newton 21/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 5.003e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.727738071 | proteklo 19.9 s
Newton 22/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 5.538e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.727665348 | proteklo 20.8 s
Newton 23/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 4.909e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.727595302 | proteklo 21.8 s
Newton 24/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 5.478e-03
Najbolji cilj: 16.727527114
Max|gradijent|: 5.478e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.767315230

loto7_4712_k87_loto_plus_1734.csv | ridge 0.10000 | validacija 2
SEED: 39 | uzorci: 1213 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.733321434 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.733321434 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 1.0022 | radijus 5.0000e-01 | Max|grad| 1.215e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.732068980 | proteklo 1.8 s
Newton 2/24 | prihvacen | rho 0.9873 | radijus 5.0000e-01 | Max|grad| 1.389e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731772338 | proteklo 2.8 s
Newton 3/24 | prihvacen | rho 1.0037 | radijus 5.0000e-01 | Max|grad| 6.162e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731581727 | proteklo 3.8 s
Newton 4/24 | prihvacen | rho 0.9935 | radijus 5.0000e-01 | Max|grad| 1.062e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.731427248 | proteklo 4.8 s
Newton 5/24 | prihvacen | rho 1.0019 | radijus 5.0000e-01 | Max|grad| 4.431e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.731274942 | proteklo 5.8 s
Newton 6/24 | prihvacen | rho 0.9888 | radijus 5.0000e-01 | Max|grad| 1.132e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.731121452 | proteklo 6.7 s
Newton 7/24 | prihvacen | rho 1.0016 | radijus 5.0000e-01 | Max|grad| 4.381e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.730959661 | proteklo 7.7 s
Newton 8/24 | prihvacen | rho 0.9858 | radijus 5.0000e-01 | Max|grad| 1.173e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.730796581 | proteklo 8.7 s
Newton 9/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 4.357e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.730628361 | proteklo 9.6 s
Newton 10/24 | prihvacen | rho 0.9852 | radijus 5.0000e-01 | Max|grad| 1.157e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.730467122 | proteklo 10.6 s
Newton 11/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 4.071e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.730309651 | proteklo 11.6 s
Newton 12/24 | prihvacen | rho 0.9857 | radijus 5.0000e-01 | Max|grad| 1.074e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.730167585 | proteklo 12.6 s
Newton 13/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 3.584e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.730037053 | proteklo 13.6 s
Newton 14/24 | prihvacen | rho 0.9867 | radijus 5.0000e-01 | Max|grad| 9.480e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.729924996 | proteklo 14.5 s
Newton 15/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 3.032e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.729826637 | proteklo 15.5 s
Newton 16/24 | prihvacen | rho 0.9877 | radijus 5.0000e-01 | Max|grad| 8.208e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.729743708 | proteklo 16.5 s
Newton 17/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 2.557e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.729671582 | proteklo 17.5 s
Newton 18/24 | prihvacen | rho 0.9877 | radijus 5.0000e-01 | Max|grad| 7.302e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.729609018 | proteklo 18.4 s
Newton 19/24 | prihvacen | rho 1.0017 | radijus 5.0000e-01 | Max|grad| 2.234e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.729552407 | proteklo 19.4 s
Newton 20/24 | prihvacen | rho 0.9856 | radijus 5.0000e-01 | Max|grad| 6.939e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.729499891 | proteklo 20.4 s
Newton 21/24 | prihvacen | rho 1.0020 | radijus 5.0000e-01 | Max|grad| 2.068e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.729448630 | proteklo 21.4 s
Newton 22/24 | prihvacen | rho 0.9814 | radijus 5.0000e-01 | Max|grad| 7.121e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.729397404 | proteklo 22.4 s
Newton 23/24 | prihvacen | rho 1.0022 | radijus 5.0000e-01 | Max|grad| 2.092e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.729343200 | proteklo 23.4 s
Newton 24/24 | prihvacen | rho 0.9757 | radijus 5.0000e-01 | Max|grad| 7.803e-03
Najbolji cilj: 16.729285760
Max|gradijent|: 7.803e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.737404734

IZBOR REGULARIZACIJE
Ridge 0.01000 | validacioni NLL 16.751598894
Ridge 0.10000 | validacioni NLL 16.752403080
Izabran ridge: 0.01000
Pri validacionoj razlici do 0.000010 NLL prednost ima jaca regularizacija.

loto7_4712_k87_loto_plus_1734.csv | odvojeni test
SEED: 39 | uzorci: 1386 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.727443458 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.727443458 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 0.9983 | radijus 5.0000e-01 | Max|grad| 1.397e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.725980452 | proteklo 1.9 s
Newton 2/24 | prihvacen | rho 0.9110 | radijus 5.0000e-01 | Max|grad| 2.953e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.725054004 | proteklo 2.9 s
Newton 3/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 1.422e-02
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.723926934 | proteklo 3.9 s
Newton 4/24 | prihvacen | rho 0.8293 | radijus 5.0000e-01 | Max|grad| 3.754e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.722793502 | proteklo 4.9 s
Newton 5/24 | prihvacen | rho 0.9969 | radijus 5.0000e-01 | Max|grad| 1.526e-02
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.720922793 | proteklo 5.9 s
Newton 6/24 | prihvacen | rho 0.7021 | radijus 5.0000e-01 | Max|grad| 4.686e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.719365339 | proteklo 6.9 s
Newton 7/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 1.488e-02
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.716174029 | proteklo 7.9 s
Newton 8/24 | prihvacen | rho 0.4022 | radijus 5.0000e-01 | Max|grad| 1.025e-01
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.714849896 | proteklo 8.9 s
Newton 9/24 | prihvacen | rho 1.0063 | radijus 5.0000e-01 | Max|grad| 1.336e-02
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.709685954 | proteklo 9.9 s
Newton 10/24 | prihvacen | rho 0.7297 | radijus 5.0000e-01 | Max|grad| 7.402e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.707524027 | proteklo 10.9 s
Newton 11/24 | prihvacen | rho 1.0132 | radijus 5.0000e-01 | Max|grad| 1.124e-02
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.704389717 | proteklo 11.9 s
Newton 12/24 | prihvacen | rho 0.9093 | radijus 5.0000e-01 | Max|grad| 4.208e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.702583970 | proteklo 12.9 s
Newton 13/24 | prihvacen | rho 1.0096 | radijus 5.0000e-01 | Max|grad| 1.149e-02
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.700918988 | proteklo 13.9 s
Newton 14/24 | prihvacen | rho 0.9753 | radijus 5.0000e-01 | Max|grad| 3.343e-02
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.699693286 | proteklo 14.9 s
Newton 15/24 | prihvacen | rho 1.0089 | radijus 5.0000e-01 | Max|grad| 1.322e-02
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.698568367 | proteklo 15.9 s
Newton 16/24 | prihvacen | rho 0.9804 | radijus 5.0000e-01 | Max|grad| 2.703e-02
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.697744230 | proteklo 17.0 s
Newton 17/24 | prihvacen | rho 1.0075 | radijus 5.0000e-01 | Max|grad| 1.369e-02
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.696930230 | proteklo 17.9 s
Newton 18/24 | prihvacen | rho 1.0027 | radijus 5.0000e-01 | Max|grad| 2.026e-02
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.696313874 | proteklo 18.9 s
Newton 19/24 | prihvacen | rho 1.0017 | radijus 5.0000e-01 | Max|grad| 1.122e-02
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.695773864 | proteklo 19.9 s
Newton 20/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 1.732e-02
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.695313791 | proteklo 20.9 s
Newton 21/24 | prihvacen | rho 1.0024 | radijus 5.0000e-01 | Max|grad| 8.651e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.694920004 | proteklo 21.9 s
Newton 22/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 1.452e-02
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.694589589 | proteklo 23.0 s
Newton 23/24 | prihvacen | rho 1.0021 | radijus 5.0000e-01 | Max|grad| 6.855e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.694309563 | proteklo 24.0 s
Newton 24/24 | prihvacen | rho 1.0038 | radijus 5.0000e-01 | Max|grad| 1.245e-02
Najbolji cilj: 16.694070027
Max|gradijent|: 1.245e-02
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

Odvojeni test NLL: 16.709201893
Test ne menja izabranu regularizaciju.

loto7_4712_k87_loto_plus_1734.csv | zavrsni trening cele istorije
SEED: 39 | uzorci: 1733 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.697283212 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.697283212 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 1.0060 | radijus 5.0000e-01 | Max|grad| 1.091e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.696195573 | proteklo 2.0 s
Newton 2/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 1.158e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.695807563 | proteklo 3.0 s
Newton 3/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 8.202e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.695542641 | proteklo 4.1 s
Newton 4/24 | prihvacen | rho 1.0024 | radijus 5.0000e-01 | Max|grad| 9.253e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.695338123 | proteklo 5.2 s
Newton 5/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 6.653e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.695168269 | proteklo 6.2 s
Newton 6/24 | prihvacen | rho 1.0028 | radijus 5.0000e-01 | Max|grad| 8.680e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.695016411 | proteklo 7.3 s
Newton 7/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 6.001e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.694877774 | proteklo 8.3 s
Newton 8/24 | prihvacen | rho 1.0029 | radijus 5.0000e-01 | Max|grad| 8.539e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.694746670 | proteklo 9.4 s
Newton 9/24 | prihvacen | rho 0.9969 | radijus 5.0000e-01 | Max|grad| 5.689e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.694621729 | proteklo 10.4 s
Newton 10/24 | prihvacen | rho 1.0027 | radijus 5.0000e-01 | Max|grad| 8.472e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.694500620 | proteklo 11.5 s
Newton 11/24 | prihvacen | rho 0.9965 | radijus 5.0000e-01 | Max|grad| 5.531e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.694382808 | proteklo 12.6 s
Newton 12/24 | prihvacen | rho 1.0024 | radijus 5.0000e-01 | Max|grad| 8.435e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.694267231 | proteklo 13.6 s
Newton 13/24 | prihvacen | rho 0.9961 | radijus 5.0000e-01 | Max|grad| 5.445e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.694153667 | proteklo 14.7 s
Newton 14/24 | prihvacen | rho 1.0021 | radijus 5.0000e-01 | Max|grad| 8.471e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.694041571 | proteklo 15.8 s
Newton 15/24 | prihvacen | rho 0.9957 | radijus 5.0000e-01 | Max|grad| 5.394e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.693930880 | proteklo 17.0 s
Newton 16/24 | prihvacen | rho 1.0018 | radijus 5.0000e-01 | Max|grad| 8.563e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.693821246 | proteklo 18.1 s
Newton 17/24 | prihvacen | rho 0.9954 | radijus 5.0000e-01 | Max|grad| 5.360e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.693712715 | proteklo 19.2 s
Newton 18/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 8.639e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.693604958 | proteklo 20.2 s
Newton 19/24 | prihvacen | rho 0.9951 | radijus 5.0000e-01 | Max|grad| 5.337e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.693498122 | proteklo 21.3 s
Newton 20/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 8.702e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.693391777 | proteklo 22.4 s
Newton 21/24 | prihvacen | rho 0.9949 | radijus 5.0000e-01 | Max|grad| 5.326e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.693286175 | proteklo 23.4 s
Newton 22/24 | prihvacen | rho 1.0012 | radijus 5.0000e-01 | Max|grad| 8.759e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.693180701 | proteklo 24.5 s
Newton 23/24 | prihvacen | rho 0.9946 | radijus 5.0000e-01 | Max|grad| 5.331e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.693075723 | proteklo 25.5 s
Newton 24/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 8.823e-03
Najbolji cilj: 16.692970377
Max|gradijent|: 8.823e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

PUNI HESIJAN: zavrseno 0/36 kolona.
Hesijan: 3/36 kolona.
Hesijan: 6/36 kolona.
Hesijan: 9/36 kolona.
Hesijan: 12/36 kolona.
Hesijan: 15/36 kolona.
Hesijan: 18/36 kolona.
Hesijan: 21/36 kolona.
Hesijan: 24/36 kolona.
Hesijan: 27/36 kolona.
Hesijan: 30/36 kolona.
Hesijan: 33/36 kolona.
Hesijan: 36/36 kolona.

Racunam eksplicitne NEXT Jakobijane.

REZULTAT: loto7_4712_k87_loto_plus_1734.csv
SEED: 39
NEXT: 1 x 13 y 32 z 36
Izabran ridge: 0.01000
Validacioni NLL: 16.751598894
Odvojeni test NLL: 16.709201893
Modelna verovatnoca NEXT: 9.57190969644e-07
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Maksimum je tacan za naucenu proizvodnu raspodelu.
Izbor najvecih naucenih Bornovih tezina ovde sledi iz oblika zajednicke raspodele; nisu koriscene istorijske ucestalosti.
Greska Qiskit provere: 6.939e-18
Osetljivost na istoriju: 1.676755806e-01
Zavrsni Max|gradijent|: 8.823e-03
Optimizacioni proracuni: 120
Od toga Hesijan-vektor: 96
EKSPPLICITNE MATRICE
Jakobijan amplituda: (64, 36)
Jakobijan Bornovih tezina: (39, 36)
Jakobijan verovatnoca ukljucivanja: (39, 36)
Puni Hesijan regularizovanog NLL-a: (36, 36)
Rang NEXT Jakobijana ukljucivanja: 36
Greska simetrije Hesijana: 2.220e-14
Najmanja sopstvena vrednost Hesijana: -1.478863963e-02
Najveca sopstvena vrednost Hesijana: 9.004665480e+00
Matrice i dijagnostika: /Users/4c/Desktop/GHQ/Astra/Quantum_Jacobians_Hessians_Newton/Quantum_Jacobians_Hessians_Newton_V2_checkpoints/diagnostics_2747e6c8feb90637467c41d37348ab6d8ef130621d4b438c1ec7126fcfa3771b.npz
Mali gradijent i numericka zakrivljenost nisu dokaz globalnog optimuma.
Test histogram pogodaka:
0: 67
1: 143
2: 94
3: 42
4: 1
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.328530
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.230767540
 2: 0.145867250
 3: 0.269021778
 4: 0.127465264
 5: 0.161392477
 6: 0.161559961
 7: 0.210896200
 8: 0.161716534
 9: 0.171467701
10: 0.172862975
11: 0.148461936
12: 0.158703053
13: 0.287688195
14: 0.187438175
15: 0.171279918
16: 0.170551019
17: 0.159582125
18: 0.145578096
19: 0.160889059
20: 0.138377044
21: 0.156821273
22: 0.201468361
23: 0.178687557
24: 0.138473321
25: 0.158284398
26: 0.255663005
27: 0.179569427
28: 0.158815708
29: 0.177905778
30: 0.202177170
31: 0.161916814
32: 0.258293780
33: 0.160778427
34: 0.211204342
35: 0.153755823
36: 0.218036649
37: 0.166947698
38: 0.154894064
39: 0.164740102
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
DIJAGNOSTIKA: Zavrsni Hesijan ima negativnu zakrivljenost; lokalni minimum nije potvrdjen.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
ZAVRSNI NEXT REZULTATI
loto7_4712_k87.csv | NEXT: 6 x 14 y 27 z 30 | seed: 39 | ridge: 0.10000 | test NLL: 16.761194127
loto7_4712_k87_loto_2978.csv | NEXT: 6 x 15 y 29 z 35 | seed: 39 | ridge: 0.10000 | test NLL: 16.737695029
loto7_4712_k87_loto_plus_1734.csv | NEXT: 1 x 13 y 32 z 36 | seed: 39 | ridge: 0.01000 | test NLL: 16.709201893
"""



"""
V2 sa aktivnim SEED = 39, bez uvođenja slučajnog generisanja. 
SEED = 39 određuje početne uglove i deo je identiteta kontrolnih tačaka. 
Uklonio sam računanje q*q iz izvoda, jer kod veoma malih Bornovih težina može numerički postati nula.
Dodao sam i eksplicitne matrice za završnu proveru: Jakobijan amplituda, Jakobijan verovatnoća uključenja i puni Hesijan funkcije cilja. 
Trening i dalje koristi proizvode Hesijana i vektora radi manjeg troška; puni Hesijan računa se jednom za završni model svakog CSV-a.
"""
