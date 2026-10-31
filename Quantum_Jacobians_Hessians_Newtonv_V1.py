# -*- coding: utf-8 -*-

# Quantum_Jacobians_Hessians_Newton


"""
Quantum Jacobians Hessians Newton V1 — Loto 7/39

Osnovna ideja:
ucenje kvantne raspodele pomocu izvoda prvog i drugog reda.

Qiskit:
- sest kubita po registru;
- sedam nezavisno pripremljenih identicnih registara;
- zbog istih ulaza i parametara simulira se jedan registar;
- rezultat se matematicki koristi za sedam nezavisnih kopija.

Izmedju sedam registara nema kvantnih kapija.
Unutar registra postoje RX, RZZ i RY kapije.

Raspodela kombinacije:
    P(S | istorija) = product_{j in S} q_j / e_7(q)

q_j su Bornove tezine dobijene Qiskit evolucijom.
e_7 je elementarni simetricni polinom sedmog stepena.

To odgovara uslovljavanju sedam nezavisnih izlaza na:
- brojeve od 1 do 39;
- sedam razlicitih brojeva;
- zanemarivanje redosleda izvlacenja.

Nema dodatnog naucenog sprezanja izmedju buducih brojeva
osim zajednickog uslova da kombinacija ima sedam elemenata.

Izvodi:
- Jakobijan se koristi preko njegovog dejstva na vektor;
- gradijent se racuna adjungovanim postupkom;
- proizvod Hesijana i vektora dobija se diferenciranjem
  kvantne evolucije, normalizacije i adjungovanog postupka;
- cela velika Hesijanova matrica ne formira se u treningu.

Newtonov korak:
truncated conjugate gradient unutar oblasti poverenja.

Qiskit racuna amplitude.
Izvodi, normalizacija i optimizacija su klasicni numericki
proracuni, bez klasicnog prediktivnog regresora.

Nema nasumicnog generisanja ni uzorkovanja merenja.
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


VERSION = "Quantum_Jacobians_Hessians_Newton_V1_20261031"

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

INITIAL_RADIUS = 0.5
MAX_RADIUS = 2.0
MIN_RADIUS = 1.0e-7

ACCEPTANCE_RATIO = 0.10
GRADIENT_TOLERANCE = 1.0e-5

SCRIPT_DIR = (
    Path(__file__).resolve().parent
    if "__file__" in globals()
    else Path.cwd()
)

CACHE_DIR = SCRIPT_DIR / "Quantum_Newton_V1_checkpoints"


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
        """
        Vraca:
        logZ, marginalne verovatnoce,
        smerni izvod logZ, smerne izvode marginala.

        direction je smerni izvod log_weights.
        """

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

        dprefix = np.zeros_like(prefix) if directional else None
        dsuffix = np.zeros_like(suffix) if directional else None

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
        dmarginal = np.empty_like(x) if directional else None

        for j in range(m):
            terms = (
                prefix[:, j, :k]
                + suffix[:, j + 1, k - 1::-1]
            )

            leave_one = np.logaddexp.reduce(terms, axis=1)

            marginal[:, j] = np.exp(
                x[:, j] + leave_one - log_z
            )

            if directional:
                dterms = (
                    dprefix[:, j, :k]
                    + dsuffix[:, j + 1, k - 1::-1]
                )

                probabilities = np.exp(
                    terms - leave_one[:, None]
                )

                dleave_one = np.sum(
                    probabilities * dterms,
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
            "Marginalne verovatnoce nisu u dozvoljenom opsegu.",
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
            targets.shape == (len(log_weights), self.picks)
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

        finite(result, "Logaritmi verovatnoca kombinacija")

        require(
            np.max(result) <= 2.0e-8,
            "Verovatnoca kombinacije prelazi jedan.",
        )

        return result

    def maximum(self, log_weights):
        """
        Za proizvod pozitivnih tezina, uz fiksnu kardinalnost,
        globalni maksimum cine najvece naucene tezine.
        To nisu istorijske ucestalosti.
        """

        values = np.asarray(log_weights, dtype=np.float64)

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

        log_z = self.statistics(values[None, :])[0][0]

        log_probability = float(
            values[selected].sum() - log_z
        )

        require(
            -math.log(
                math.comb(self.numbers, self.picks)
            ) - 2.0e-8
            <= log_probability
            <= 2.0e-8,
            "Neispravna verovatnoca globalnog maksimuma.",
        )

        return selected, log_probability

    def independent_log_z(self, log_weights):
        values = np.asarray(
            log_weights, dtype=np.longdouble
        )

        shift = np.max(values)
        weights = np.exp(values - shift)

        coefficients = np.zeros(
            self.picks + 1, dtype=np.longdouble
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
            "Dimenzija kvantnog ulaza nije stepen dvojke.",
        )

        finite(self.inputs, "Ulazne amplitude")

        require(
            np.max(
                np.abs(
                    np.sum(
                        np.abs(self.inputs) ** 2, axis=1
                    ) - 1.0
                )
            ) < 2.0e-10,
            "Ulazne amplitude nisu normalizovane.",
        )

        self.padded = next_power_of_two(self.samples)

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

    def seed(self):
        positions = np.arange(
            self.size, dtype=np.float64
        )

        return (
            0.08 * np.sin(positions + 0.3)
            + 0.03 * np.cos(0.7 * positions)
        )

    def operations(self, parameters):
        values = np.asarray(parameters).reshape(
            self.layers, 3, self.qubits
        )

        operations = []

        for layer in range(self.layers):
            base = layer * 3 * self.qubits

            for q in range(self.qubits):
                operations.append(
                    (
                        "rx",
                        float(values[layer, 0, q]),
                        (q,),
                        base + q,
                    )
                )

            for q in range(self.qubits):
                operations.append(
                    (
                        "rzz",
                        float(values[layer, 1, q]),
                        (q, (q + 1) % self.qubits),
                        base + self.qubits + q,
                    )
                )

            for q in range(self.qubits):
                operations.append(
                    (
                        "ry",
                        float(values[layer, 2, q]),
                        (q,),
                        base + 2 * self.qubits + q,
                    )
                )

        return operations

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
                matrix * (1.0 - 2.0 * parity)[None, :]
            ).reshape(-1)

        raise RuntimeError("Nepoznat generator.")

    def forward(self, parameters, batch, direction=None):
        parameters = np.asarray(
            parameters, dtype=np.float64
        )

        require(
            parameters.shape == (self.size,)
            and batch.dimension == self.dimension,
            "Neispravni parametri ili ulaz kvantnog izvora.",
        )

        finite(parameters, "Parametri kvantnog kola")

        if direction is None:
            state = Statevector(batch.packed).evolve(
                self.circuit(parameters)
            )

            tangent = None

        else:
            direction = np.asarray(
                direction, dtype=np.float64
            )

            require(
                direction.shape == parameters.shape,
                "Neispravan smer u prostoru parametara.",
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

        packed = np.asarray(state.data).reshape(
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
            "Kvantna evolucija nije sacuvala normu.",
        )

        if batch.padded > batch.samples:
            require(
                np.max(
                    np.abs(packed[batch.samples:])
                ) < 2.0e-12,
                "Promenjene su prazne indeksne grane.",
            )

        tangent_data = (
            None
            if tangent is None
            else np.asarray(tangent.data)
        )

        return (
            np.asarray(state.data),
            tangent_data,
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
            finite(hvp, "Proizvod Hesijana i vektora")

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

        require(
            np.all(q > 0.0),
            "Kandidat ima nultu Bornovu tezinu.",
        )

        finite(q, "Bornove tezine")

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

        log_probability = target_log_weight - log_z

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

        adjoint = np.zeros(
            (self.batch.padded, dimension),
            dtype=np.complex128,
        )

        adjoint[:n, :m] = (
            residual * packed[:n, :m] / q
        )

        if directional:
            dadjoint = np.zeros_like(adjoint)

            dadjoint[:n, :m] = (
                (
                    dmarginal / q
                    - residual * dq / (q * q)
                )
                * packed[:n, :m]
                + residual
                * dpacked[:n, :m] / q
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

        regularization = 0.5 * self.ridge * float(
            np.mean(parameters ** 2)
        )

        gradient += (
            self.ridge * parameters / self.source.size
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
    c = float(np.dot(position, position) - radius * radius)

    require(a > 0.0, "Nulti pravac pri granici oblasti poverenja.")

    discriminant = max(0.0, b * b - 4.0 * a * c)

    tau = (-b + math.sqrt(discriminant)) / (2.0 * a)

    require(
        tau >= -1.0e-10,
        "Neispravan presek granice oblasti poverenja.",
    )

    return max(0.0, tau)


def truncated_newton_step(gradient, radius, hessian_product):
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

    reason = "Dostignut budzet Krylov koraka."

    for _ in range(MAX_CG_STEPS):
        h_direction = hessian_product(direction)

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

        if math.sqrt(new_squared_residual) <= 0.1 * initial_norm:
            return (
                position,
                h_position,
                "Krylov rezidual je dovoljno mali.",
            )

        beta = (
            new_squared_residual / squared_residual
        )

        direction = -new_residual + beta * direction

        residual = new_residual
        squared_residual = new_squared_residual

    return position, h_position, reason


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
        source.seed()
        if warm is None
        else warm.parameters.copy()
    )

    key = digest_items(
        VERSION,
        label,
        source.qubits,
        source.layers,
        ridge,
        MAX_ITERATIONS,
        MAX_ORACLE_CALLS,
        MAX_CG_STEPS,
        INITIAL_RADIUS,
        MAX_RADIUS,
        GRADIENT_TOLERANCE,
        qiskit.__version__,
        np.__version__,
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
                    str(saved["version"].item()) == VERSION,
                    "Neodgovarajuca verzija kontrolne tacke.",
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

            finite(parameters, "Sacuvani trenutni parametri")
            finite(best_parameters, "Sacuvani najbolji parametri")

            require(
                0.0 < radius <= MAX_RADIUS
                and 0 <= iterations <= MAX_ITERATIONS
                and 0 <= calls <= MAX_ORACLE_CALLS
                and 0 <= hessian_calls <= calls,
                "Neispravno sacuvano stanje optimizatora.",
            )

            say(label, "| ucitana kontrolna tacka.")

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
        f"Uzorci: {len(inputs)} "
        f"| ridge: {ridge:.5f} "
        f"| parametri: {source.size}"
    )

    def save(done_flag, status):
        atomic_npz(
            path,
            version=np.array(VERSION),
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
            candidate, direction=direction
        )

    try:
        while not done and iterations < MAX_ITERATIONS:
            gradient_norm = float(
                np.max(np.abs(current_gradient))
            )

            if gradient_norm <= GRADIENT_TOLERANCE:
                message = "Dostignut uslov malog gradijenta."
                break

            if radius < MIN_RADIUS:
                message = "Oblast poverenja postala je vrlo mala."
                break

            iterations += 1
            old_radius = radius

            def hessian_product(direction):
                return oracle(
                    parameters, direction=direction
                )[2]

            step, h_step, step_reason = truncated_newton_step(
                current_gradient,
                radius,
                hessian_product,
            )

            finite(step, "Newtonov korak")

            require(
                np.linalg.norm(step) <= radius + 2.0e-8,
                "Newtonov korak je izvan oblasti poverenja.",
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
                message = "Kvadratni model nije dao pozitivan pad."
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
            ):
                radius *= 0.25
                message = (
                    "Numericki kandidat je odbijen; "
                    "oblast poverenja je smanjena."
                )
                save(False, message)
                continue

            if candidate_loss < best_loss:
                best_parameters = candidate.copy()
                best_loss = float(candidate_loss)
                best_gradient = candidate_gradient.copy()

            actual_reduction = current_loss - candidate_loss
            ratio = actual_reduction / predicted_reduction

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
        -log_weights, axis=1, kind="stable"
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
        1.0 + current + 0.5 * previous + 0.25 * older
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
    fingerprint = hashlib.sha256(raw).hexdigest()

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

    draws = np.asarray(rows, dtype=np.int64)

    say()
    say("ULAZ:", path.name)
    say(f"Redova: {len(draws)} | praznih redova: {blanks}")
    say("SHA256:", fingerprint)
    say("Redosled kola ostaje redosled iz CSV-a.")

    return draws


def self_checks():
    say()
    say("Pokrecem matematicke i Qiskit samoprovere.")

    distribution = FixedSizeDistribution(6, 3)

    log_weights = np.array(
        [[-0.2, 0.1, -0.7, 0.4, -0.1, 0.2]],
        dtype=np.float64,
    )

    direction = np.array(
        [[0.3, -0.1, 0.2, 0.05, -0.2, 0.1]],
        dtype=np.float64,
    )

    log_z, marginal, dlog_z, dmarginal = (
        distribution.statistics(
            log_weights, direction
        )
    )

    subsets = np.array(
        list(combinations(range(6), 3)),
        dtype=np.int64,
    )

    weights = np.exp(
        log_weights[0, subsets].sum(axis=1)
    )

    probabilities = weights / weights.sum()

    exact_marginal = np.zeros(6, dtype=np.float64)

    for subset, probability in zip(
        subsets, probabilities
    ):
        exact_marginal[subset] += probability

    require(
        abs(log_z[0] - math.log(float(weights.sum())))
        < 2.0e-12,
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
        log_weights + epsilon * direction
    )

    minus = distribution.statistics(
        log_weights - epsilon * direction
    )

    require(
        np.max(
            np.abs(
                (plus[0] - minus[0]) / (2.0 * epsilon)
                - dlog_z
            )
        ) < 2.0e-8,
        "Smerni izvod normalizacije nije ispravan.",
    )

    require(
        np.max(
            np.abs(
                (plus[1] - minus[1]) / (2.0 * epsilon)
                - dmarginal
            )
        ) < 2.0e-8,
        "Smerni izvod marginala nije ispravan.",
    )

    selected, log_probability = distribution.maximum(
        log_weights[0]
    )

    require(
        np.array_equal(
            selected,
            subsets[int(np.argmax(weights))],
        ),
        "Izbor maksimuma nije ispravan.",
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
            distribution.independent_log_z(log_weights[0])
            - log_z[0]
        ) < 2.0e-12,
        "Nezavisna normalizacija nije ispravna.",
    )

    for picks in (1, 6):
        boundary = FixedSizeDistribution(6, picks)

        value = boundary.statistics(
            np.zeros((1, 6), dtype=np.float64)
        )[0][0]

        require(
            abs(
                value - math.log(math.comb(6, picks))
            ) < 2.0e-12,
            "Granicni slucaj raspodele nije ispravan.",
        )

    source = QuantumSource(qubits=3, layers=2)

    inputs = np.empty((3, 8), dtype=np.complex128)

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

    parameters = source.seed()

    vector = np.cos(
        np.arange(source.size) + 0.7
    )

    vector /= np.linalg.norm(vector)

    loss, gradient, hvp = objective.evaluate(
        parameters, direction=vector
    )

    require(
        math.isfinite(loss),
        "Samoprovera funkcije cilja nije konacna.",
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

        error = abs(numerical - gradient[index])
        gradient_error = max(gradient_error, error)

        require(
            error
            < 2.0e-5 * (1.0 + abs(gradient[index])),
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
        np.max(np.abs(numerical_hvp - hvp))
    )

    require(
        hvp_error
        < 3.0e-5 * (
            1.0 + np.max(np.abs(hvp))
        ),
        "Hesijan puta vektor ne prolazi numericku proveru.",
    )

    hessian = np.empty(
        (source.size, source.size),
        dtype=np.float64,
    )

    for index in range(source.size):
        basis_vector = np.zeros(source.size)
        basis_vector[index] = 1.0

        hessian[:, index] = objective.evaluate(
            parameters, direction=basis_vector
        )[2]

    symmetry_error = float(
        np.max(np.abs(hessian - hessian.T))
    )

    require(
        symmetry_error < 2.0e-7,
        "Mali Hesijan nije simetrican.",
    )

    batch = QuantumBatch(inputs)

    state, tangent = source.forward(
        parameters, batch, direction=vector
    )

    plus_state = source.forward(
        parameters + epsilon * vector, batch
    )[0]

    minus_state = source.forward(
        parameters - epsilon * vector, batch
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
        "Jakobijan amplituda puta vektor nije ispravan.",
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
            "Paketna simulacija se razlikuje od pojedinacne.",
        )

    # Provera Newtonovog potproblema na poznatoj kvadratnoj funkciji.
    diagonal = np.array([1.0, 2.0, 4.0])
    small_gradient = np.array([0.2, -0.3, 0.1])

    step, h_step, _ = truncated_newton_step(
        small_gradient,
        radius=2.0,
        hessian_product=lambda v: diagonal * v,
    )

    require(
        np.max(
            np.abs(
                step + small_gradient / diagonal
            )
        ) < 2.0e-10,
        "Newtonov potproblem nije ispravno resen.",
    )

    require(
        np.max(
            np.abs(h_step - diagonal * step)
        ) < 2.0e-10,
        "Pracenje Hesijana koraka nije ispravno.",
    )

    say("Samoprovere su prosle.")
    say(f"Greska gradijenta: {gradient_error:.3e}")
    say(f"Greska Hesijan-vektor: {hvp_error:.3e}")
    say(f"Greska simetrije malog Hesijana: {symmetry_error:.3e}")
    say(f"Greska Jakobijan-vektor: {tangent_error:.3e}")
    say(f"Greska paketne Qiskit simulacije: {batch_error:.3e}")


def resource_report(loaded, source):
    largest = max(
        len(draws) for _, draws in loaded
    )

    padded = next_power_of_two(largest)

    state_bytes = padded * BASIS_SIZE * 16

    dp_bytes = (
        largest * (NUMBERS + 1) * (PICKS + 1) * 8
    )

    estimate = (
        384 * 1024**2
        + 12 * state_bytes
        + 10 * dp_bytes
        + 5 * largest * BASIS_SIZE * 16
    )

    circuit = source.circuit(source.seed())

    say()
    say("PROCENA RESURSA")
    say(f"Kubita po registru: {source.qubits}")
    say(f"Identicnih nezavisnih registara: {PICKS}")
    say(f"Naucenih uglova: {source.size}")
    say(f"Kapija jednog naucenog kola: {len(circuit.data)}")
    say(f"Dubina jednog naucenog kola: {circuit.depth()}")

    say(
        "Jedan izlaz se simulira jednom i koristi za sedam "
        "nezavisno pripremljenih identicnih registara."
    )

    say(
        "Indeksni bitovi paketne simulacije sluze "
        "samo vektorizaciji nezavisnih ulaza."
    )

    say(
        f"Konzervativna procena memorije: "
        f"{estimate / 1024**3:.2f} GiB"
    )

    say(
        "Procena nije izmereni vrh memorije procesa."
    )

    require(
        estimate <= 3.0 * 1024**3,
        "Procenjena memorija prelazi tri GiB. "
        "Trening nije zapocet.",
    )


def process_file(path, draws, source):
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
                "validation": weighted_nll / validation_rows,
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

    # Pri skoro jednakom validacionom NLL-u prednost ima
    # jaca regularizacija. Test ne ucestvuje u ovom izboru.
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
        f"Tolerancija za prednost jace regularizacije: "
        f"{VALIDATION_TIE:.6f} NLL"
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
        source, final_fit, next_input
    )[0]

    log_z, marginal, _, _ = DISTRIBUTION.statistics(
        log_weights[None, :]
    )

    selected_numbers, log_probability = (
        DISTRIBUTION.maximum(log_weights)
    )

    alternative_log_z = (
        DISTRIBUTION.independent_log_z(log_weights)
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

    direct = Statevector(next_input[0]).evolve(
        source.circuit(final_fit.parameters)
    )

    direct_weights = (
        np.abs(direct.data[:NUMBERS]) ** 2
    )

    qiskit_error = float(
        np.max(
            np.abs(
                direct_weights - np.exp(log_weights)
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
        np.max(np.ptp(recent_marginal, axis=0))
    )

    numbers = tuple(
        int(value + 1) for value in selected_numbers
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
        np.array_equal(selected_numbers, historical)
        for historical in draws
    ):
        warnings.append(
            "NEXT ponavlja istorijsku kombinaciju."
        )

    if np.array_equal(selected_numbers, draws[-1]):
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

    say()
    say("REZULTAT:", path.name)
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
        "Globalni maksimum dobijen je tacnim svojstvom "
        "proizvodne raspodele, bez generisanja kombinacija."
    )

    say(f"Greska Qiskit provere: {qiskit_error:.3e}")
    say(f"Osetljivost na istoriju: {sensitivity:.9e}")

    say(
        f"Zavrsni Max|gradijent|: "
        f"{final_fit.gradient:.3e}"
    )

    say(
        f"Zavrsni optimizacioni proracuni: "
        f"{final_fit.calls}"
    )

    say(
        f"Od toga Hesijan-vektor proracuni: "
        f"{final_fit.hessian_calls}"
    )

    say(
        "Mali gradijent sam po sebi ne dokazuje "
        "globalni minimum parametara."
    )

    say("Test histogram pogodaka:")

    for hits, frequency in enumerate(histogram):
        say(f"{hits}: {int(frequency)}")

    mean_hits = float(
        np.dot(
            np.arange(PICKS + 1), histogram
        ) / histogram.sum()
    )

    say(f"Prosecan broj pogodaka na testu: {mean_hits:.6f}")

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
            int(value + 1) for value in draws[-1]
        ),
    }


def main():
    say("Quantum Jacobians Hessians Newton V1 — Loto 7/39")
    say("Python:", sys.version.split()[0])
    say("Qiskit:", qiskit.__version__)
    say("NumPy:", np.__version__)
    say()

    say("Lokalna Qiskit simulacija kompleksnih amplituda.")

    say(
        "Sedam nezavisno pripremljenih identicnih registara "
        "od po sest kubita."
    )

    say(
        "Njihov isti sestokubitni izlaz racuna se jednom."
    )

    say(
        "Raspodela se uslovljava na sedam razlicitih "
        "brojeva od 1 do 39."
    )

    say(
        "Izmedju registara nema kvantnog sprezanja."
    )

    say(
        "Optimizacija koristi Jakobijan, gradijent "
        "i analiticki Hesijan puta vektor."
    )

    say(
        "Newtonov potproblem resava se skracenim "
        "Krylov postupkom u oblasti poverenja."
    )

    say(
        "Negativna zakrivljenost usmerava korak "
        "prema granici oblasti poverenja."
    )

    say(
        "Kriticni koraci prihvataju se prema stvarnom "
        "smanjenju funkcije cilja."
    )

    say(
        "Konacne razlike koriste se samo u samoproverama."
    )

    say(
        "Nema klasifikatora, slucajnih podela, "
        "generisanja kombinacija ni uzorkovanja merenja."
    )

    say(
        "Validacija: 60%-70% i 70%-80% istorijskih prelaza."
    )

    say(
        "Odvojeni test: poslednjih 20% prelaza."
    )

    say(
        "Test je jedan korak unapred: prethodno stvarno "
        "kolo dostupno je kao istorijski ulaz."
    )

    say(
        "Zavrsni trening koristi sve istorijske prelaze."
    )

    say(
        f"Budzet po treningu: {MAX_ITERATIONS} Newtonovih "
        f"iteracija i {MAX_ORACLE_CALLS} proracuna izvoda."
    )

    say(
        f"Najvise {MAX_CG_STEPS} Hesijan-vektor koraka "
        "po Newtonovom potproblemu."
    )

    say(
        "Ucitavanje i zavrsna provera kandidata rade se "
        "dodatno, izvan optimizacionog brojaca."
    )

    say(
        "Dostignut budzet ne odbacuje najbolji "
        "numericki vazeci kandidat."
    )

    loaded = []

    for path in FILES:
        loaded.append(
            (path, load_draws(path))
        )

    source = QuantumSource()

    resource_report(loaded, source)
    self_checks()

    CACHE_DIR.mkdir(
        parents=True, exist_ok=True
    )

    say()
    say("Kontrolne tacke:", str(CACHE_DIR))

    results = []

    for path, draws in loaded:
        results.append(
            process_file(path, draws, source)
        )

    say()
    say("=" * 70)
    say("ZAVRSNI NEXT REZULTATI")

    for result in results:
        say(
            f"{result['name']} | NEXT: "
            + " ".join(map(str, result["next"]))
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
            "Najbolji kandidat i stanje Newtonovog "
            "postupka sacuvani su tokom treninga."
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
Quantum Jacobians Hessians Newton V1 — Loto 7/39
Python: 3.11.13
Qiskit: 1.4.4
NumPy: 1.26.4

Lokalna Qiskit simulacija kompleksnih amplituda.
Sedam nezavisno pripremljenih identicnih registara od po sest kubita.
Njihov isti sestokubitni izlaz racuna se jednom.
Raspodela se uslovljava na sedam razlicitih brojeva od 1 do 39.
Izmedju registara nema kvantnog sprezanja.
Optimizacija koristi Jakobijan, gradijent i analiticki Hesijan puta vektor.
Newtonov potproblem resava se skracenim Krylov postupkom u oblasti poverenja.
Negativna zakrivljenost usmerava korak prema granici oblasti poverenja.
Kriticni koraci prihvataju se prema stvarnom smanjenju funkcije cilja.
Konacne razlike koriste se samo u samoproverama.
Nema klasifikatora, slucajnih podela, generisanja kombinacija ni uzorkovanja merenja.
Validacija: 60%-70% i 70%-80% istorijskih prelaza.
Odvojeni test: poslednjih 20% prelaza.
Test je jedan korak unapred: prethodno stvarno kolo dostupno je kao istorijski ulaz.
Zavrsni trening koristi sve istorijske prelaze.
Budzet po treningu: 24 Newtonovih iteracija i 120 proracuna izvoda.
Najvise 4 Hesijan-vektor koraka po Newtonovom potproblemu.
Ucitavanje i zavrsna provera kandidata rade se dodatno, izvan optimizacionog brojaca.
Dostignut budzet ne odbacuje najbolji numericki vazeci kandidat.

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
Kubita po registru: 6
Identicnih nezavisnih registara: 7
Naucenih uglova: 36
Kapija jednog naucenog kola: 36
Dubina jednog naucenog kola: 16
Jedan izlaz se simulira jednom i koristi za sedam nezavisno pripremljenih identicnih registara.
Indeksni bitovi paketne simulacije sluze samo vektorizaciji nezavisnih ulaza.
Konzervativna procena memorije: 0.60 GiB
Procena nije izmereni vrh memorije procesa.

Pokrecem matematicke i Qiskit samoprovere.
Samoprovere su prosle.
Greska gradijenta: 3.317e-10
Greska Hesijan-vektor: 3.346e-10
Greska simetrije malog Hesijana: 8.188e-16
Greska Jakobijan-vektor: 5.424e-11
Greska paketne Qiskit simulacije: 2.223e-16

Kontrolne tacke: /Quantum_Jacobians_Hessians_Newton/Quantum_Newton_V1_checkpoints

======================================================================
OBRADA: loto7_4712_k87.csv
Istorijskih prelaza: 4711
loto7_4712_k87.csv | ridge 0.01000 | validacija 1 | ucitana kontrolna tacka.

loto7_4712_k87.csv | ridge 0.01000 | validacija 1
Uzorci: 2826 | ridge: 0.01000 | parametri: 36
Proracun 55/120 | Hesijan-vektor 45 | najbolji cilj 16.746709399 | proteklo 1.0 s
Newton 12/24 | prihvacen | rho 0.9405 | radijus 5.0000e-01 | Max|grad| 1.209e-02
Proracun 60/120 | Hesijan-vektor 49 | najbolji cilj 16.746535545 | proteklo 3.0 s
Newton 13/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 3.301e-03
Proracun 65/120 | Hesijan-vektor 53 | najbolji cilj 16.746389015 | proteklo 5.1 s
Newton 14/24 | prihvacen | rho 0.9770 | radijus 5.0000e-01 | Max|grad| 9.971e-03
Proracun 70/120 | Hesijan-vektor 57 | najbolji cilj 16.746273692 | proteklo 7.1 s
Newton 15/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 2.664e-03
Proracun 75/120 | Hesijan-vektor 61 | najbolji cilj 16.746176633 | proteklo 9.2 s
Newton 16/24 | prihvacen | rho 0.9886 | radijus 5.0000e-01 | Max|grad| 8.869e-03
Proracun 80/120 | Hesijan-vektor 65 | najbolji cilj 16.746092196 | proteklo 11.2 s
Newton 17/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 2.400e-03
Proracun 85/120 | Hesijan-vektor 69 | najbolji cilj 16.746017168 | proteklo 13.2 s
Newton 18/24 | prihvacen | rho 0.9884 | radijus 5.0000e-01 | Max|grad| 7.776e-03
Proracun 90/120 | Hesijan-vektor 73 | najbolji cilj 16.745952901 | proteklo 15.3 s
Newton 19/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 2.126e-03
Proracun 95/120 | Hesijan-vektor 77 | najbolji cilj 16.745896705 | proteklo 17.3 s
Newton 20/24 | prihvacen | rho 0.9906 | radijus 5.0000e-01 | Max|grad| 6.735e-03
Proracun 100/120 | Hesijan-vektor 81 | najbolji cilj 16.745849088 | proteklo 19.3 s
Newton 21/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.874e-03
Proracun 105/120 | Hesijan-vektor 85 | najbolji cilj 16.745807925 | proteklo 21.4 s
Newton 22/24 | prihvacen | rho 0.9941 | radijus 5.0000e-01 | Max|grad| 5.843e-03
Proracun 110/120 | Hesijan-vektor 89 | najbolji cilj 16.745772730 | proteklo 23.4 s
Newton 23/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.646e-03
Proracun 115/120 | Hesijan-vektor 93 | najbolji cilj 16.745742203 | proteklo 25.4 s
Newton 24/24 | prihvacen | rho 0.9963 | radijus 5.0000e-01 | Max|grad| 5.103e-03
Najbolji cilj: 16.745715776
Max|gradijent|: 5.103e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.759487648

loto7_4712_k87.csv | ridge 0.01000 | validacija 2
Uzorci: 3297 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.747693114 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.747693114 | proteklo 1.9 s
Newton 1/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 6.030e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.747407960 | proteklo 4.0 s
Newton 2/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 4.657e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.747339564 | proteklo 6.2 s
Newton 3/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.801e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.747312460 | proteklo 8.3 s
Newton 4/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 2.648e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.747297100 | proteklo 10.4 s
Newton 5/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.347e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.747286347 | proteklo 12.6 s
Newton 6/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 2.212e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747277396 | proteklo 14.8 s
Newton 7/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.216e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747269627 | proteklo 16.9 s
Newton 8/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 2.051e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747262470 | proteklo 19.0 s
Newton 9/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.150e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747255777 | proteklo 21.1 s
Newton 10/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 1.982e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747249362 | proteklo 23.2 s
Newton 11/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.111e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747243157 | proteklo 25.3 s
Newton 12/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 1.962e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747237064 | proteklo 27.4 s
Newton 13/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.091e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747231035 | proteklo 29.6 s
Newton 14/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 1.979e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747225001 | proteklo 31.7 s
Newton 15/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.084e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747218917 | proteklo 33.7 s
Newton 16/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 2.028e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747212724 | proteklo 35.8 s
Newton 17/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.091e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747206372 | proteklo 37.9 s
Newton 18/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 2.109e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747199808 | proteklo 40.0 s
Newton 19/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.111e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747192970 | proteklo 42.0 s
Newton 20/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 2.222e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747185805 | proteklo 44.1 s
Newton 21/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.144e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747178233 | proteklo 46.2 s
Newton 22/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 2.370e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747170198 | proteklo 48.3 s
Newton 23/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.192e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747161594 | proteklo 50.3 s
Newton 24/24 | prihvacen | rho 0.9973 | radijus 5.0000e-01 | Max|grad| 2.557e-03
Najbolji cilj: 16.747152358
Max|gradijent|: 2.557e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.755790209

loto7_4712_k87.csv | ridge 0.10000 | validacija 1
Uzorci: 2826 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.894949754 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0447 | radijus 5.0000e-01 | Max|grad| 5.274e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.752859674 | proteklo 1.6 s
Newton 2/24 | prihvacen | rho 1.0104 | radijus 5.0000e-01 | Max|grad| 1.022e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.750008038 | proteklo 3.6 s
Newton 3/24 | prihvacen | rho 1.0118 | radijus 5.0000e-01 | Max|grad| 2.002e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.748998129 | proteklo 5.6 s
Newton 4/24 | prihvacen | rho 1.0023 | radijus 5.0000e-01 | Max|grad| 4.319e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.748542730 | proteklo 7.6 s
Newton 5/24 | prihvacen | rho 0.9179 | radijus 5.0000e-01 | Max|grad| 1.356e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.748269827 | proteklo 9.6 s
Newton 6/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 4.146e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.748037944 | proteklo 11.6 s
Newton 7/24 | prihvacen | rho 0.9578 | radijus 5.0000e-01 | Max|grad| 1.315e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747825112 | proteklo 13.6 s
Newton 8/24 | prihvacen | rho 1.0012 | radijus 5.0000e-01 | Max|grad| 4.578e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747603809 | proteklo 15.6 s
Newton 9/24 | prihvacen | rho 0.9377 | radijus 5.0000e-01 | Max|grad| 1.344e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747401308 | proteklo 17.6 s
Newton 10/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 4.331e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747196376 | proteklo 19.6 s
Newton 11/24 | prihvacen | rho 0.9342 | radijus 5.0000e-01 | Max|grad| 1.171e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747036067 | proteklo 21.6 s
Newton 12/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 3.299e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.746896284 | proteklo 23.6 s
Newton 13/24 | prihvacen | rho 0.9637 | radijus 5.0000e-01 | Max|grad| 9.084e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.746795101 | proteklo 25.6 s
Newton 14/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 2.258e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746715035 | proteklo 27.6 s
Newton 15/24 | prihvacen | rho 0.9896 | radijus 5.0000e-01 | Max|grad| 7.734e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746648523 | proteklo 29.6 s
Newton 16/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 1.976e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.746591567 | proteklo 31.6 s
Newton 17/24 | prihvacen | rho 0.9897 | radijus 5.0000e-01 | Max|grad| 6.786e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.746542528 | proteklo 33.6 s
Newton 18/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 1.727e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.746499527 | proteklo 35.6 s
Newton 19/24 | prihvacen | rho 0.9891 | radijus 5.0000e-01 | Max|grad| 5.871e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.746463162 | proteklo 37.6 s
Newton 20/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.498e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746431783 | proteklo 39.7 s
Newton 21/24 | prihvacen | rho 0.9916 | radijus 5.0000e-01 | Max|grad| 5.016e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746405535 | proteklo 41.7 s
Newton 22/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.295e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746383179 | proteklo 43.7 s
Newton 23/24 | prihvacen | rho 0.9942 | radijus 5.0000e-01 | Max|grad| 4.295e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746364360 | proteklo 45.7 s
Newton 24/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 1.122e-03
Najbolji cilj: 16.746348285
Max|gradijent|: 1.122e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.758793510

loto7_4712_k87.csv | ridge 0.10000 | validacija 2
Uzorci: 3297 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.748210246 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.748210246 | proteklo 1.9 s
Newton 1/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 5.807e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.747948499 | proteklo 4.0 s
Newton 2/24 | prihvacen | rho 1.0046 | radijus 5.0000e-01 | Max|grad| 5.452e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.747881855 | proteklo 6.1 s
Newton 3/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.756e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.747858234 | proteklo 8.1 s
Newton 4/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 1.891e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.747846528 | proteklo 10.2 s
Newton 5/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.029e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.747838482 | proteklo 12.3 s
Newton 6/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.577e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747832050 | proteklo 14.4 s
Newton 7/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 9.359e-04
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747826615 | proteklo 16.5 s
Newton 8/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 1.429e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747821708 | proteklo 18.6 s
Newton 9/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 8.868e-04
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747817219 | proteklo 20.6 s
Newton 10/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.350e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747812974 | proteklo 22.7 s
Newton 11/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 8.578e-04
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747808936 | proteklo 24.8 s
Newton 12/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.307e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747805016 | proteklo 26.9 s
Newton 13/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 8.427e-04
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747801200 | proteklo 28.9 s
Newton 14/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 1.289e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747797423 | proteklo 31.0 s
Newton 15/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 8.391e-04
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747793682 | proteklo 33.1 s
Newton 16/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 1.290e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747789924 | proteklo 35.1 s
Newton 17/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 8.453e-04
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747786148 | proteklo 37.2 s
Newton 18/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 1.308e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747782309 | proteklo 39.3 s
Newton 19/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 8.605e-04
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747778408 | proteklo 41.4 s
Newton 20/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 1.341e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747774400 | proteklo 43.4 s
Newton 21/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 8.840e-04
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747770287 | proteklo 45.5 s
Newton 22/24 | prihvacen | rho 1.0013 | radijus 5.0000e-01 | Max|grad| 1.387e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747766025 | proteklo 47.6 s
Newton 23/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 9.157e-04
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747761617 | proteklo 49.7 s
Newton 24/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 1.445e-03
Najbolji cilj: 16.747757013
Max|gradijent|: 1.445e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.753926033

IZBOR REGULARIZACIJE
Ridge 0.01000 | validacioni NLL 16.757638928
Ridge 0.10000 | validacioni NLL 16.756359772
Izabran ridge: 0.10000
Tolerancija za prednost jace regularizacije: 0.000010 NLL

loto7_4712_k87.csv | odvojeni test
Uzorci: 3768 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.748600675 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.748600675 | proteklo 1.9 s
Newton 1/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 4.306e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.748379873 | proteklo 4.1 s
Newton 2/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 4.490e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.748310385 | proteklo 6.2 s
Newton 3/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 2.445e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.748272032 | proteklo 8.3 s
Newton 4/24 | prihvacen | rho 1.0012 | radijus 5.0000e-01 | Max|grad| 3.407e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.748245081 | proteklo 10.5 s
Newton 5/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 2.065e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.748224031 | proteklo 12.6 s
Newton 6/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 3.034e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.748205984 | proteklo 14.8 s
Newton 7/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.918e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748189760 | proteklo 16.9 s
Newton 8/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 2.891e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.748174554 | proteklo 19.1 s
Newton 9/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 1.861e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.748159914 | proteklo 21.2 s
Newton 10/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 2.862e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.748145510 | proteklo 23.4 s
Newton 11/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 1.855e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.748131069 | proteklo 25.6 s
Newton 12/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 2.910e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.748116406 | proteklo 27.7 s
Newton 13/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 1.885e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.748101288 | proteklo 29.9 s
Newton 14/24 | prihvacen | rho 0.9977 | radijus 5.0000e-01 | Max|grad| 3.023e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.748085576 | proteklo 32.0 s
Newton 15/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 1.949e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.748069017 | proteklo 34.1 s
Newton 16/24 | prihvacen | rho 0.9969 | radijus 5.0000e-01 | Max|grad| 3.200e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.748051486 | proteklo 36.3 s
Newton 17/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 2.045e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.748032665 | proteklo 38.4 s
Newton 18/24 | prihvacen | rho 0.9960 | radijus 5.0000e-01 | Max|grad| 3.446e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.748012420 | proteklo 40.6 s
Newton 19/24 | prihvacen | rho 1.0012 | radijus 5.0000e-01 | Max|grad| 2.177e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747990323 | proteklo 42.7 s
Newton 20/24 | prihvacen | rho 0.9949 | radijus 5.0000e-01 | Max|grad| 3.773e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747966210 | proteklo 44.9 s
Newton 21/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 2.349e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747939476 | proteklo 47.0 s
Newton 22/24 | prihvacen | rho 0.9935 | radijus 5.0000e-01 | Max|grad| 4.203e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747909905 | proteklo 49.1 s
Newton 23/24 | prihvacen | rho 1.0015 | radijus 5.0000e-01 | Max|grad| 2.570e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747876603 | proteklo 51.3 s
Newton 24/24 | prihvacen | rho 0.9919 | radijus 5.0000e-01 | Max|grad| 4.762e-03
Najbolji cilj: 16.747839279
Max|gradijent|: 4.762e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

Odvojeni test NLL: 16.758623715
Test ne menja izabranu regularizaciju.

loto7_4712_k87.csv | zavrsni trening cele istorije
Uzorci: 4711 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.750115664 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.750115664 | proteklo 3.7 s
Newton 1/24 | prihvacen | rho 1.0139 | radijus 5.0000e-01 | Max|grad| 1.557e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.749764966 | proteklo 7.8 s
Newton 2/24 | prihvacen | rho 0.9843 | radijus 5.0000e-01 | Max|grad| 1.175e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.749592569 | proteklo 11.9 s
Newton 3/24 | prihvacen | rho 1.0033 | radijus 5.0000e-01 | Max|grad| 5.856e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.749468799 | proteklo 16.0 s
Newton 4/24 | prihvacen | rho 0.9795 | radijus 5.0000e-01 | Max|grad| 1.168e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749363557 | proteklo 20.1 s
Newton 5/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 5.095e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749247978 | proteklo 24.1 s
Newton 6/24 | prihvacen | rho 0.9619 | radijus 5.0000e-01 | Max|grad| 1.261e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.749122212 | proteklo 28.2 s
Newton 7/24 | prihvacen | rho 1.0029 | radijus 5.0000e-01 | Max|grad| 5.555e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748965689 | proteklo 32.2 s
Newton 8/24 | prihvacen | rho 0.9432 | radijus 5.0000e-01 | Max|grad| 1.349e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.748791344 | proteklo 36.3 s
Newton 9/24 | prihvacen | rho 1.0034 | radijus 5.0000e-01 | Max|grad| 6.606e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.748559263 | proteklo 40.3 s
Newton 10/24 | prihvacen | rho 0.9164 | radijus 5.0000e-01 | Max|grad| 1.430e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.748301736 | proteklo 44.4 s
Newton 11/24 | prihvacen | rho 1.0033 | radijus 5.0000e-01 | Max|grad| 7.761e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747938449 | proteklo 48.5 s
Newton 12/24 | prihvacen | rho 0.8580 | radijus 5.0000e-01 | Max|grad| 1.853e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747550297 | proteklo 52.5 s
Newton 13/24 | prihvacen | rho 1.0032 | radijus 5.0000e-01 | Max|grad| 8.469e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746952844 | proteklo 56.6 s
Newton 14/24 | prihvacen | rho 0.7398 | radijus 5.0000e-01 | Max|grad| 3.161e-02
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746433002 | proteklo 60.6 s
Newton 15/24 | prihvacen | rho 1.0047 | radijus 5.0000e-01 | Max|grad| 9.819e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.745494030 | proteklo 64.6 s
Newton 16/24 | prihvacen | rho 0.5787 | radijus 5.0000e-01 | Max|grad| 4.792e-02
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.744975039 | proteklo 68.7 s
Newton 17/24 | prihvacen | rho 1.0042 | radijus 5.0000e-01 | Max|grad| 9.177e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.743805245 | proteklo 72.8 s
Newton 18/24 | prihvacen | rho 0.4966 | radijus 5.0000e-01 | Max|grad| 5.011e-02
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.743402293 | proteklo 76.8 s
Newton 19/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 7.639e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.742442926 | proteklo 80.9 s
Newton 20/24 | prihvacen | rho 0.9788 | radijus 5.0000e-01 | Max|grad| 1.379e-02
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.742152989 | proteklo 84.9 s
Newton 21/24 | prihvacen | rho 0.9892 | radijus 5.0000e-01 | Max|grad| 7.046e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.741903599 | proteklo 89.0 s
Newton 22/24 | prihvacen | rho 0.9758 | radijus 5.0000e-01 | Max|grad| 1.152e-02
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.741705820 | proteklo 93.0 s
Newton 23/24 | prihvacen | rho 0.9909 | radijus 5.0000e-01 | Max|grad| 7.211e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.741531552 | proteklo 97.1 s
Newton 24/24 | prihvacen | rho 0.9829 | radijus 5.0000e-01 | Max|grad| 8.233e-03
Najbolji cilj: 16.741403523
Max|gradijent|: 8.233e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

REZULTAT: loto7_4712_k87.csv
NEXT: 6 9 13 19 27 29 30
Izabran ridge: 0.10000
Validacioni NLL: 16.756359772
Odvojeni test NLL: 16.758623715
Modelna verovatnoca NEXT: 1.04027193088e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 3.553e-15
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Globalni maksimum dobijen je tacnim svojstvom proizvodne raspodele, bez generisanja kombinacija.
Greska Qiskit provere: 6.939e-18
Osetljivost na istoriju: 1.743697071e-01
Zavrsni Max|gradijent|: 8.233e-03
Zavrsni optimizacioni proracuni: 120
Od toga Hesijan-vektor proracuni: 96
Mali gradijent sam po sebi ne dokazuje globalni minimum parametara.
Test histogram pogodaka:
0: 220
1: 382
2: 239
3: 95
4: 7
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.243902
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.200010823
 2: 0.146361160
 3: 0.172840251
 4: 0.143923582
 5: 0.194456478
 6: 0.240421317
 7: 0.207517438
 8: 0.152860359
 9: 0.239925600
10: 0.137289918
11: 0.143272987
12: 0.140911402
13: 0.240101584
14: 0.238973286
15: 0.172991574
16: 0.144691597
17: 0.163647172
18: 0.169398535
19: 0.249287218
20: 0.138346270
21: 0.190499982
22: 0.192245586
23: 0.170863748
24: 0.153549709
25: 0.136239514
26: 0.164118707
27: 0.253665672
28: 0.153782581
29: 0.245691521
30: 0.272488666
31: 0.151493113
32: 0.163220274
33: 0.156565852
34: 0.162895423
35: 0.143580139
36: 0.162935351
37: 0.173175527
38: 0.154742005
39: 0.161018080
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4712_k87_loto_2978.csv
Istorijskih prelaza: 2977

loto7_4712_k87_loto_2978.csv | ridge 0.01000 | validacija 1
Uzorci: 1786 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.896534961 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0467 | radijus 5.0000e-01 | Max|grad| 4.956e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.754218368 | proteklo 0.8 s
Newton 2/24 | prihvacen | rho 1.0096 | radijus 5.0000e-01 | Max|grad| 1.522e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.751425214 | proteklo 1.8 s
Newton 3/24 | prihvacen | rho 0.8770 | radijus 5.0000e-01 | Max|grad| 2.472e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.750274479 | proteklo 2.9 s
Newton 4/24 | prihvacen | rho 1.0037 | radijus 5.0000e-01 | Max|grad| 4.837e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.749594708 | proteklo 3.9 s
Newton 5/24 | odbijen | rho 0.0366 | radijus 1.2500e-01 | Max|grad| 4.837e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749561779 | proteklo 4.9 s
Newton 6/24 | prihvacen | rho 0.8410 | radijus 2.5000e-01 | Max|grad| 1.857e-02
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.749082389 | proteklo 6.0 s
Newton 7/24 | prihvacen | rho 1.0051 | radijus 2.5000e-01 | Max|grad| 8.883e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.748555163 | proteklo 7.0 s
Newton 8/24 | prihvacen | rho 0.8768 | radijus 2.5000e-01 | Max|grad| 1.763e-02
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748152836 | proteklo 8.0 s
Newton 9/24 | prihvacen | rho 1.0018 | radijus 2.5000e-01 | Max|grad| 4.511e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747810783 | proteklo 9.1 s
Newton 10/24 | prihvacen | rho 0.9218 | radijus 2.5000e-01 | Max|grad| 1.088e-02
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747618365 | proteklo 10.1 s
Newton 11/24 | prihvacen | rho 0.9993 | radijus 2.5000e-01 | Max|grad| 3.591e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747472124 | proteklo 11.2 s
Newton 12/24 | prihvacen | rho 0.9936 | radijus 2.5000e-01 | Max|grad| 1.007e-02
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747340890 | proteklo 12.3 s
Newton 13/24 | prihvacen | rho 0.9983 | radijus 2.5000e-01 | Max|grad| 5.435e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747204493 | proteklo 13.4 s
Newton 14/24 | prihvacen | rho 0.9963 | radijus 2.5000e-01 | Max|grad| 1.004e-02
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747069473 | proteklo 14.5 s
Newton 15/24 | prihvacen | rho 0.9968 | radijus 2.5000e-01 | Max|grad| 7.071e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746932401 | proteklo 15.5 s
Newton 16/24 | prihvacen | rho 0.9942 | radijus 2.5000e-01 | Max|grad| 8.683e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.746813493 | proteklo 16.6 s
Newton 17/24 | prihvacen | rho 0.9988 | radijus 2.5000e-01 | Max|grad| 3.894e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.746709781 | proteklo 17.6 s
Newton 18/24 | prihvacen | rho 0.9847 | radijus 2.5000e-01 | Max|grad| 7.076e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.746628336 | proteklo 18.6 s
Newton 19/24 | prihvacen | rho 0.9996 | radijus 2.5000e-01 | Max|grad| 2.634e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.746557857 | proteklo 19.7 s
Newton 20/24 | prihvacen | rho 0.9952 | radijus 2.5000e-01 | Max|grad| 6.517e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746496426 | proteklo 20.7 s
Newton 21/24 | prihvacen | rho 0.9998 | radijus 2.5000e-01 | Max|grad| 2.421e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746439568 | proteklo 21.8 s
Newton 22/24 | prihvacen | rho 0.9967 | radijus 2.5000e-01 | Max|grad| 6.308e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746385249 | proteklo 22.8 s
Newton 23/24 | prihvacen | rho 0.9998 | radijus 2.5000e-01 | Max|grad| 2.418e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746332195 | proteklo 23.8 s
Newton 24/24 | prihvacen | rho 0.9942 | radijus 2.5000e-01 | Max|grad| 6.327e-03
Najbolji cilj: 16.746279671
Max|gradijent|: 6.327e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.750087383

loto7_4712_k87_loto_2978.csv | ridge 0.01000 | validacija 2
Uzorci: 2083 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.746831391 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.746831391 | proteklo 1.8 s
Newton 1/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 7.143e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.746140952 | proteklo 3.7 s
Newton 2/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 7.359e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.746011614 | proteklo 5.7 s
Newton 3/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 3.506e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.745945541 | proteklo 7.6 s
Newton 4/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 4.335e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.745899281 | proteklo 9.5 s
Newton 5/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 3.227e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.745856237 | proteklo 11.4 s
Newton 6/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 3.751e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.745815644 | proteklo 13.3 s
Newton 7/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 3.222e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.745775635 | proteklo 15.2 s
Newton 8/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 3.821e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.745736060 | proteklo 17.1 s
Newton 9/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 3.283e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.745696544 | proteklo 19.1 s
Newton 10/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 3.902e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.745656990 | proteklo 21.0 s
Newton 11/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 3.359e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.745617201 | proteklo 22.9 s
Newton 12/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 3.998e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.745577017 | proteklo 24.8 s
Newton 13/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 3.443e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.745536391 | proteklo 26.7 s
Newton 14/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 4.100e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.745495144 | proteklo 28.6 s
Newton 15/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 3.523e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.745453343 | proteklo 30.5 s
Newton 16/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 4.196e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.745410848 | proteklo 32.4 s
Newton 17/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 3.589e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.745367801 | proteklo 34.4 s
Newton 18/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 4.274e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.745324142 | proteklo 36.3 s
Newton 19/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 3.633e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.745280077 | proteklo 38.2 s
Newton 20/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 4.321e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.745235647 | proteklo 40.1 s
Newton 21/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 3.643e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.745191109 | proteklo 42.0 s
Newton 22/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 4.325e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.745146603 | proteklo 43.9 s
Newton 23/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 3.613e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.745102408 | proteklo 45.9 s
Newton 24/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 4.278e-03
Najbolji cilj: 16.745058728
Max|gradijent|: 4.278e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.761447389

loto7_4712_k87_loto_2978.csv | ridge 0.10000 | validacija 1
Uzorci: 1786 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.896710112 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0466 | radijus 5.0000e-01 | Max|grad| 4.951e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.754311453 | proteklo 0.8 s
Newton 2/24 | prihvacen | rho 1.0096 | radijus 5.0000e-01 | Max|grad| 1.540e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.751512132 | proteklo 1.8 s
Newton 3/24 | prihvacen | rho 0.9071 | radijus 5.0000e-01 | Max|grad| 2.335e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.750383661 | proteklo 2.9 s
Newton 4/24 | prihvacen | rho 1.0035 | radijus 5.0000e-01 | Max|grad| 4.961e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.749767562 | proteklo 3.9 s
Newton 5/24 | prihvacen | rho 0.5977 | radijus 5.0000e-01 | Max|grad| 2.337e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.749427797 | proteklo 4.9 s
Newton 6/24 | prihvacen | rho 1.0043 | radijus 5.0000e-01 | Max|grad| 6.830e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.748742568 | proteklo 6.0 s
Newton 7/24 | prihvacen | rho 0.9505 | radijus 5.0000e-01 | Max|grad| 1.326e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.748414798 | proteklo 7.0 s
Newton 8/24 | prihvacen | rho 1.0013 | radijus 5.0000e-01 | Max|grad| 6.085e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.748188187 | proteklo 8.0 s
Newton 9/24 | prihvacen | rho 0.9578 | radijus 5.0000e-01 | Max|grad| 9.265e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.748039883 | proteklo 9.1 s
Newton 10/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 3.610e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747930660 | proteklo 10.1 s
Newton 11/24 | prihvacen | rho 0.9831 | radijus 5.0000e-01 | Max|grad| 7.742e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747841983 | proteklo 11.1 s
Newton 12/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 3.146e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747761978 | proteklo 12.2 s
Newton 13/24 | prihvacen | rho 0.9930 | radijus 5.0000e-01 | Max|grad| 8.142e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747678729 | proteklo 13.2 s
Newton 14/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 3.329e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747590969 | proteklo 14.3 s
Newton 15/24 | prihvacen | rho 0.9895 | radijus 5.0000e-01 | Max|grad| 8.389e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747501733 | proteklo 15.5 s
Newton 16/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 3.238e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747411755 | proteklo 16.5 s
Newton 17/24 | prihvacen | rho 0.9869 | radijus 5.0000e-01 | Max|grad| 7.891e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747328497 | proteklo 17.6 s
Newton 18/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 2.949e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747251593 | proteklo 18.6 s
Newton 19/24 | prihvacen | rho 0.9884 | radijus 5.0000e-01 | Max|grad| 6.892e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747185739 | proteklo 19.7 s
Newton 20/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 2.616e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747128657 | proteklo 20.7 s
Newton 21/24 | prihvacen | rho 0.9937 | radijus 5.0000e-01 | Max|grad| 5.858e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747080857 | proteklo 21.7 s
Newton 22/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 2.355e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747039858 | proteklo 22.8 s
Newton 23/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 5.124e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747004095 | proteklo 23.8 s
Newton 24/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 2.222e-03
Najbolji cilj: 16.746972138
Max|gradijent|: 2.222e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.750414264

loto7_4712_k87_loto_2978.csv | ridge 0.10000 | validacija 2
Uzorci: 2083 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.747537338 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.747537338 | proteklo 1.7 s
Newton 1/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 3.228e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.746821669 | proteklo 3.7 s
Newton 2/24 | prihvacen | rho 1.0050 | radijus 5.0000e-01 | Max|grad| 4.625e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.746728892 | proteklo 5.6 s
Newton 3/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 2.428e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.746686178 | proteklo 7.6 s
Newton 4/24 | prihvacen | rho 1.0025 | radijus 5.0000e-01 | Max|grad| 4.055e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.746650918 | proteklo 9.5 s
Newton 5/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 2.255e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.746618723 | proteklo 11.4 s
Newton 6/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 3.898e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.746587732 | proteklo 13.3 s
Newton 7/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 2.176e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.746557762 | proteklo 15.3 s
Newton 8/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 3.812e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.746528418 | proteklo 17.2 s
Newton 9/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 2.104e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.746499721 | proteklo 19.1 s
Newton 10/24 | prihvacen | rho 0.9976 | radijus 5.0000e-01 | Max|grad| 3.716e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.746471714 | proteklo 21.0 s
Newton 11/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 2.021e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.746444411 | proteklo 22.9 s
Newton 12/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 3.620e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.746417903 | proteklo 24.8 s
Newton 13/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 1.931e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.746392131 | proteklo 26.8 s
Newton 14/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 3.551e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.746367104 | proteklo 28.7 s
Newton 15/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.898e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.746342697 | proteklo 30.6 s
Newton 16/24 | prihvacen | rho 0.9981 | radijus 5.0000e-01 | Max|grad| 3.505e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.746318878 | proteklo 32.5 s
Newton 17/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.886e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.746295482 | proteklo 34.4 s
Newton 18/24 | prihvacen | rho 0.9981 | radijus 5.0000e-01 | Max|grad| 3.486e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.746272493 | proteklo 36.3 s
Newton 19/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.890e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.746249742 | proteklo 38.2 s
Newton 20/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 3.490e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.746227231 | proteklo 40.1 s
Newton 21/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.903e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.746204821 | proteklo 42.0 s
Newton 22/24 | prihvacen | rho 0.9976 | radijus 5.0000e-01 | Max|grad| 3.506e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.746182535 | proteklo 43.9 s
Newton 23/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 1.917e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.746160276 | proteklo 45.9 s
Newton 24/24 | prihvacen | rho 0.9973 | radijus 5.0000e-01 | Max|grad| 3.524e-03
Najbolji cilj: 16.746138100
Max|gradijent|: 3.524e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.760059956

IZBOR REGULARIZACIJE
Ridge 0.01000 | validacioni NLL 16.755776933
Ridge 0.10000 | validacioni NLL 16.755245215
Izabran ridge: 0.10000
Tolerancija za prednost jace regularizacije: 0.000010 NLL

loto7_4712_k87_loto_2978.csv | odvojeni test
Uzorci: 2381 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.747970708 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.747970708 | proteklo 1.8 s
Newton 1/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 6.751e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.747526512 | proteklo 3.7 s
Newton 2/24 | prihvacen | rho 0.9799 | radijus 5.0000e-01 | Max|grad| 8.338e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.747421885 | proteklo 5.7 s
Newton 3/24 | prihvacen | rho 0.9997 | radijus 5.0000e-01 | Max|grad| 2.977e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.747377970 | proteklo 7.6 s
Newton 4/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 5.130e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.747351638 | proteklo 9.6 s
Newton 5/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.868e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.747330742 | proteklo 11.6 s
Newton 6/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 3.837e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.747312856 | proteklo 13.6 s
Newton 7/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.641e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.747296927 | proteklo 15.6 s
Newton 8/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 2.521e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.747282826 | proteklo 17.7 s
Newton 9/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.507e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.747269944 | proteklo 19.7 s
Newton 10/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 2.358e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.747258192 | proteklo 21.7 s
Newton 11/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.416e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.747247248 | proteklo 23.7 s
Newton 12/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 2.232e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.747237025 | proteklo 25.6 s
Newton 13/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.382e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.747227360 | proteklo 27.6 s
Newton 14/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 2.120e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.747218181 | proteklo 29.5 s
Newton 15/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.364e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.747209394 | proteklo 31.5 s
Newton 16/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 2.035e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.747200947 | proteklo 33.5 s
Newton 17/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.353e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.747192771 | proteklo 35.5 s
Newton 18/24 | prihvacen | rho 0.9992 | radijus 5.0000e-01 | Max|grad| 1.972e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.747184830 | proteklo 37.4 s
Newton 19/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 1.349e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.747177069 | proteklo 39.4 s
Newton 20/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 1.932e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.747169469 | proteklo 41.4 s
Newton 21/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.350e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.747161984 | proteklo 43.3 s
Newton 22/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.910e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.747154607 | proteklo 45.2 s
Newton 23/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 1.355e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.747147298 | proteklo 47.2 s
Newton 24/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 1.897e-03
Najbolji cilj: 16.747140060
Max|gradijent|: 1.897e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

Odvojeni test NLL: 16.738173616
Test ne menja izabranu regularizaciju.

loto7_4712_k87_loto_2978.csv | zavrsni trening cele istorije
Uzorci: 2977 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.745506534 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.745506534 | proteklo 1.8 s
Newton 1/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 5.695e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.745138763 | proteklo 3.9 s
Newton 2/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 4.634e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.745074709 | proteklo 5.9 s
Newton 3/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 2.856e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.745039931 | proteklo 7.9 s
Newton 4/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 3.681e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.745013281 | proteklo 9.9 s
Newton 5/24 | prihvacen | rho 0.9987 | radijus 5.0000e-01 | Max|grad| 2.476e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.744991173 | proteklo 12.0 s
Newton 6/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 3.248e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.744971884 | proteklo 14.0 s
Newton 7/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 2.231e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.744954745 | proteklo 16.0 s
Newton 8/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 2.965e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.744939166 | proteklo 18.0 s
Newton 9/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 2.064e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.744924855 | proteklo 20.1 s
Newton 10/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 2.769e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.744911490 | proteklo 22.1 s
Newton 11/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.948e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.744898914 | proteklo 24.1 s
Newton 12/24 | prihvacen | rho 1.0001 | radijus 5.0000e-01 | Max|grad| 2.634e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.744886922 | proteklo 26.2 s
Newton 13/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.869e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.744875423 | proteklo 28.2 s
Newton 14/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 2.543e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.744864275 | proteklo 30.2 s
Newton 15/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.817e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.744853426 | proteklo 32.3 s
Newton 16/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 2.484e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.744842776 | proteklo 34.3 s
Newton 17/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.784e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.744832297 | proteklo 36.3 s
Newton 18/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 2.448e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.744821916 | proteklo 38.4 s
Newton 19/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 1.767e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.744811624 | proteklo 40.4 s
Newton 20/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 2.429e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.744801363 | proteklo 42.4 s
Newton 21/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 1.760e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.744791136 | proteklo 44.5 s
Newton 22/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 2.420e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.744780899 | proteklo 46.5 s
Newton 23/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 1.762e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.744770663 | proteklo 48.5 s
Newton 24/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 2.418e-03
Najbolji cilj: 16.744760393
Max|gradijent|: 2.418e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

REZULTAT: loto7_4712_k87_loto_2978.csv
NEXT: 6 x 15 y 29 z 35
Izabran ridge: 0.10000
Validacioni NLL: 16.755245215
Odvojeni test NLL: 16.738173616
Modelna verovatnoca NEXT: 1.81696447415e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Globalni maksimum dobijen je tacnim svojstvom proizvodne raspodele, bez generisanja kombinacija.
Greska Qiskit provere: 6.939e-18
Osetljivost na istoriju: 1.590236006e-01
Zavrsni Max|gradijent|: 2.418e-03
Zavrsni optimizacioni proracuni: 120
Od toga Hesijan-vektor proracuni: 96
Mali gradijent sam po sebi ne dokazuje globalni minimum parametara.
Test histogram pogodaka:
0: 129
1: 242
2: 167
3: 50
4: 8
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.271812
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.161686440
 2: 0.152694909
 3: 0.151177697
 4: 0.143348721
 5: 0.148430589
 6: 0.287067701
 7: 0.142585796
 8: 0.153647856
 9: 0.282575274
10: 0.162508592
11: 0.155185847
12: 0.142501789
13: 0.161591454
14: 0.216579069
15: 0.247485069
16: 0.145605299
17: 0.160985522
18: 0.150356102
19: 0.216394464
20: 0.154490765
21: 0.157488436
22: 0.259161614
23: 0.159669364
24: 0.128146942
25: 0.173665314
26: 0.157731397
27: 0.207610037
28: 0.139287108
29: 0.274476276
30: 0.185964794
31: 0.148818177
32: 0.251152951
33: 0.146924260
34: 0.168906552
35: 0.253358372
36: 0.148868192
37: 0.159187299
38: 0.181641805
39: 0.161042156
DIJAGNOSTIKA: NEXT ponavlja istorijsku kombinaciju.
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
OBRADA: loto7_4712_k87_loto_plus_1734.csv
Istorijskih prelaza: 1733

loto7_4712_k87_loto_plus_1734.csv | ridge 0.01000 | validacija 1
Uzorci: 1039 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.881386206 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0360 | radijus 5.0000e-01 | Max|grad| 5.013e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.734217086 | proteklo 0.7 s
Newton 2/24 | prihvacen | rho 1.0066 | radijus 5.0000e-01 | Max|grad| 7.470e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.731890577 | proteklo 1.6 s
Newton 3/24 | prihvacen | rho 1.1013 | radijus 5.0000e-01 | Max|grad| 1.193e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731308435 | proteklo 2.6 s
Newton 4/24 | prihvacen | rho 0.9975 | radijus 5.0000e-01 | Max|grad| 9.364e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731045232 | proteklo 3.5 s
Newton 5/24 | prihvacen | rho 0.9564 | radijus 5.0000e-01 | Max|grad| 1.533e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.730720170 | proteklo 4.5 s
Newton 6/24 | prihvacen | rho 0.9950 | radijus 5.0000e-01 | Max|grad| 9.331e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.730339934 | proteklo 5.4 s
Newton 7/24 | prihvacen | rho 0.9518 | radijus 5.0000e-01 | Max|grad| 1.389e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.730011536 | proteklo 6.3 s
Newton 8/24 | prihvacen | rho 0.9999 | radijus 5.0000e-01 | Max|grad| 7.986e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.729727957 | proteklo 7.2 s
Newton 9/24 | prihvacen | rho 0.9743 | radijus 5.0000e-01 | Max|grad| 1.312e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.729475242 | proteklo 8.2 s
Newton 10/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 7.821e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.729224280 | proteklo 9.1 s
Newton 11/24 | prihvacen | rho 0.9751 | radijus 5.0000e-01 | Max|grad| 1.301e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.728985501 | proteklo 10.0 s
Newton 12/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 6.899e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.728756822 | proteklo 10.9 s
Newton 13/24 | prihvacen | rho 0.9774 | radijus 5.0000e-01 | Max|grad| 1.199e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.728555606 | proteklo 11.9 s
Newton 14/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 5.852e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.728372903 | proteklo 12.8 s
Newton 15/24 | prihvacen | rho 0.9812 | radijus 5.0000e-01 | Max|grad| 1.108e-02
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.728211282 | proteklo 13.7 s
Newton 16/24 | prihvacen | rho 1.0009 | radijus 5.0000e-01 | Max|grad| 5.039e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.728062109 | proteklo 14.7 s
Newton 17/24 | prihvacen | rho 0.9841 | radijus 5.0000e-01 | Max|grad| 1.061e-02
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.727924249 | proteklo 15.6 s
Newton 18/24 | prihvacen | rho 1.0008 | radijus 5.0000e-01 | Max|grad| 4.391e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.727791753 | proteklo 16.6 s
Newton 19/24 | prihvacen | rho 0.9875 | radijus 5.0000e-01 | Max|grad| 1.049e-02
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.727663708 | proteklo 17.5 s
Newton 20/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 4.027e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.727536083 | proteklo 18.6 s
Newton 21/24 | prihvacen | rho 0.9913 | radijus 5.0000e-01 | Max|grad| 1.060e-02
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.727408776 | proteklo 19.6 s
Newton 22/24 | prihvacen | rho 1.0004 | radijus 5.0000e-01 | Max|grad| 4.180e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.727278793 | proteklo 20.6 s
Newton 23/24 | prihvacen | rho 0.9945 | radijus 5.0000e-01 | Max|grad| 1.090e-02
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.727146214 | proteklo 21.5 s
Newton 24/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 4.437e-03
Najbolji cilj: 16.727007625
Max|gradijent|: 4.437e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.771860507

loto7_4712_k87_loto_plus_1734.csv | ridge 0.01000 | validacija 2
Uzorci: 1213 | ridge: 0.01000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.733449795 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.733449795 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 1.0011 | radijus 5.0000e-01 | Max|grad| 1.734e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.732012722 | proteklo 1.9 s
Newton 2/24 | prihvacen | rho 0.9969 | radijus 5.0000e-01 | Max|grad| 1.923e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731646137 | proteklo 3.0 s
Newton 3/24 | prihvacen | rho 0.9998 | radijus 5.0000e-01 | Max|grad| 7.470e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731400827 | proteklo 3.9 s
Newton 4/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 9.700e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.731190122 | proteklo 4.9 s
Newton 5/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 7.158e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.730988087 | proteklo 6.0 s
Newton 6/24 | prihvacen | rho 0.9968 | radijus 5.0000e-01 | Max|grad| 1.003e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.730782706 | proteklo 6.9 s
Newton 7/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 7.345e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.730571608 | proteklo 7.9 s
Newton 8/24 | prihvacen | rho 0.9940 | radijus 5.0000e-01 | Max|grad| 1.040e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.730353728 | proteklo 8.9 s
Newton 9/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 7.385e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.730132436 | proteklo 9.9 s
Newton 10/24 | prihvacen | rho 0.9922 | radijus 5.0000e-01 | Max|grad| 1.043e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.729911774 | proteklo 10.8 s
Newton 11/24 | prihvacen | rho 0.9982 | radijus 5.0000e-01 | Max|grad| 7.138e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.729695519 | proteklo 11.9 s
Newton 12/24 | prihvacen | rho 0.9918 | radijus 5.0000e-01 | Max|grad| 9.992e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.729490005 | proteklo 12.9 s
Newton 13/24 | prihvacen | rho 0.9984 | radijus 5.0000e-01 | Max|grad| 6.601e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.729297831 | proteklo 13.9 s
Newton 14/24 | prihvacen | rho 0.9926 | radijus 5.0000e-01 | Max|grad| 9.118e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.729124148 | proteklo 14.9 s
Newton 15/24 | prihvacen | rho 0.9986 | radijus 5.0000e-01 | Max|grad| 5.881e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.728969015 | proteklo 15.8 s
Newton 16/24 | prihvacen | rho 0.9944 | radijus 5.0000e-01 | Max|grad| 8.023e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.728833852 | proteklo 16.8 s
Newton 17/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 5.155e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.728716581 | proteklo 17.8 s
Newton 18/24 | prihvacen | rho 0.9968 | radijus 5.0000e-01 | Max|grad| 6.992e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.728615175 | proteklo 18.8 s
Newton 19/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 4.574e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.728527112 | proteklo 19.7 s
Newton 20/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 6.223e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.728448893 | proteklo 20.7 s
Newton 21/24 | prihvacen | rho 0.9990 | radijus 5.0000e-01 | Max|grad| 4.210e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.728378732 | proteklo 21.6 s
Newton 22/24 | prihvacen | rho 1.0016 | radijus 5.0000e-01 | Max|grad| 5.771e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.728313219 | proteklo 22.6 s
Newton 23/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 4.057e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.728251444 | proteklo 23.6 s
Newton 24/24 | prihvacen | rho 1.0035 | radijus 5.0000e-01 | Max|grad| 5.594e-03
Najbolji cilj: 16.728190569
Max|gradijent|: 5.594e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.738565893

loto7_4712_k87_loto_plus_1734.csv | ridge 0.10000 | validacija 1
Uzorci: 1039 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.881561356 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0360 | radijus 5.0000e-01 | Max|grad| 5.009e-02
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.734309183 | proteklo 0.7 s
Newton 2/24 | prihvacen | rho 1.0067 | radijus 5.0000e-01 | Max|grad| 7.585e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.731983819 | proteklo 1.7 s
Newton 3/24 | prihvacen | rho 1.0957 | radijus 5.0000e-01 | Max|grad| 1.173e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731412960 | proteklo 2.6 s
Newton 4/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 8.949e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731163369 | proteklo 3.6 s
Newton 5/24 | prihvacen | rho 0.9561 | radijus 5.0000e-01 | Max|grad| 1.483e-02
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.730860792 | proteklo 4.5 s
Newton 6/24 | prihvacen | rho 0.9955 | radijus 5.0000e-01 | Max|grad| 8.647e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.730511062 | proteklo 5.4 s
Newton 7/24 | prihvacen | rho 0.9506 | radijus 5.0000e-01 | Max|grad| 1.320e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.730214439 | proteklo 6.4 s
Newton 8/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 7.222e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.729962574 | proteklo 7.4 s
Newton 9/24 | prihvacen | rho 0.9747 | radijus 5.0000e-01 | Max|grad| 1.231e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.729740758 | proteklo 8.3 s
Newton 10/24 | prihvacen | rho 1.0002 | radijus 5.0000e-01 | Max|grad| 7.195e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.729519970 | proteklo 9.2 s
Newton 11/24 | prihvacen | rho 0.9754 | radijus 5.0000e-01 | Max|grad| 1.240e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.729305655 | proteklo 10.2 s
Newton 12/24 | prihvacen | rho 1.0003 | radijus 5.0000e-01 | Max|grad| 6.454e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.729096606 | proteklo 11.1 s
Newton 13/24 | prihvacen | rho 0.9766 | radijus 5.0000e-01 | Max|grad| 1.147e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.728911144 | proteklo 12.1 s
Newton 14/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 5.478e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.728741980 | proteklo 13.0 s
Newton 15/24 | prihvacen | rho 0.9805 | radijus 5.0000e-01 | Max|grad| 1.052e-02
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.728593057 | proteklo 14.0 s
Newton 16/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 4.710e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.728456222 | proteklo 15.0 s
Newton 17/24 | prihvacen | rho 0.9834 | radijus 5.0000e-01 | Max|grad| 9.969e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.728330642 | proteklo 15.9 s
Newton 18/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 4.119e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.728210783 | proteklo 16.9 s
Newton 19/24 | prihvacen | rho 0.9862 | radijus 5.0000e-01 | Max|grad| 9.720e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.728096477 | proteklo 17.8 s
Newton 20/24 | prihvacen | rho 1.0007 | radijus 5.0000e-01 | Max|grad| 3.631e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.727984058 | proteklo 18.7 s
Newton 21/24 | prihvacen | rho 0.9897 | radijus 5.0000e-01 | Max|grad| 9.632e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.727874128 | proteklo 19.6 s
Newton 22/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 3.925e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.727763951 | proteklo 20.6 s
Newton 23/24 | prihvacen | rho 0.9932 | radijus 5.0000e-01 | Max|grad| 9.703e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.727653816 | proteklo 21.5 s
Newton 24/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 4.327e-03
Najbolji cilj: 16.727540984
Max|gradijent|: 4.327e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.771641102

loto7_4712_k87_loto_plus_1734.csv | ridge 0.10000 | validacija 2
Uzorci: 1213 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.733939442 | proteklo 0.0 s
Newton 1/24 | prihvacen | rho 1.0005 | radijus 5.0000e-01 | Max|grad| 4.019e-03
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.732683888 | proteklo 0.7 s
Newton 2/24 | prihvacen | rho 1.0019 | radijus 5.0000e-01 | Max|grad| 7.331e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.731999570 | proteklo 1.7 s
Newton 3/24 | prihvacen | rho 0.9748 | radijus 5.0000e-01 | Max|grad| 7.292e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.731796965 | proteklo 2.6 s
Newton 4/24 | prihvacen | rho 0.9980 | radijus 5.0000e-01 | Max|grad| 7.155e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.731644626 | proteklo 3.6 s
Newton 5/24 | prihvacen | rho 0.9980 | radijus 5.0000e-01 | Max|grad| 7.341e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.731502724 | proteklo 4.5 s
Newton 6/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 7.350e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.731359103 | proteklo 5.5 s
Newton 7/24 | prihvacen | rho 0.9979 | radijus 5.0000e-01 | Max|grad| 7.701e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.731212691 | proteklo 6.4 s
Newton 8/24 | prihvacen | rho 0.9988 | radijus 5.0000e-01 | Max|grad| 7.439e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.731063185 | proteklo 7.4 s
Newton 9/24 | prihvacen | rho 0.9973 | radijus 5.0000e-01 | Max|grad| 7.698e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.730911669 | proteklo 8.3 s
Newton 10/24 | prihvacen | rho 0.9991 | radijus 5.0000e-01 | Max|grad| 7.377e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.730760070 | proteklo 9.3 s
Newton 11/24 | prihvacen | rho 0.9967 | radijus 5.0000e-01 | Max|grad| 7.418e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.730611181 | proteklo 10.2 s
Newton 12/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 7.062e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.730467353 | proteklo 11.2 s
Newton 13/24 | prihvacen | rho 0.9966 | radijus 5.0000e-01 | Max|grad| 6.858e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.730331614 | proteklo 12.1 s
Newton 14/24 | prihvacen | rho 0.9994 | radijus 5.0000e-01 | Max|grad| 6.518e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.730205711 | proteklo 13.1 s
Newton 15/24 | prihvacen | rho 0.9970 | radijus 5.0000e-01 | Max|grad| 6.064e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.730091521 | proteklo 14.0 s
Newton 16/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 5.834e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.729989429 | proteklo 15.0 s
Newton 17/24 | prihvacen | rho 0.9978 | radijus 5.0000e-01 | Max|grad| 5.129e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.729899571 | proteklo 15.9 s
Newton 18/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 5.132e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.729821143 | proteklo 16.9 s
Newton 19/24 | prihvacen | rho 0.9989 | radijus 5.0000e-01 | Max|grad| 4.502e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.729752980 | proteklo 17.8 s
Newton 20/24 | prihvacen | rho 0.9996 | radijus 5.0000e-01 | Max|grad| 4.516e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.729693813 | proteklo 18.8 s
Newton 21/24 | prihvacen | rho 1.0000 | radijus 5.0000e-01 | Max|grad| 3.979e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.729642001 | proteklo 19.7 s
Newton 22/24 | prihvacen | rho 0.9995 | radijus 5.0000e-01 | Max|grad| 4.039e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.729596429 | proteklo 20.7 s
Newton 23/24 | prihvacen | rho 1.0010 | radijus 5.0000e-01 | Max|grad| 3.582e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.729555576 | proteklo 21.6 s
Newton 24/24 | prihvacen | rho 0.9993 | radijus 5.0000e-01 | Max|grad| 3.707e-03
Najbolji cilj: 16.729518703
Max|gradijent|: 3.707e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.
Validacioni NLL: 16.737533817

IZBOR REGULARIZACIJE
Ridge 0.01000 | validacioni NLL 16.755261175
Ridge 0.10000 | validacioni NLL 16.754636606
Izabran ridge: 0.10000
Tolerancija za prednost jace regularizacije: 0.000010 NLL

loto7_4712_k87_loto_plus_1734.csv | odvojeni test
Uzorci: 1386 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.730666563 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.730666563 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 1.0006 | radijus 5.0000e-01 | Max|grad| 9.780e-03
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.730040896 | proteklo 1.8 s
Newton 2/24 | prihvacen | rho 1.0122 | radijus 5.0000e-01 | Max|grad| 8.825e-03
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.729841832 | proteklo 2.8 s
Newton 3/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 6.921e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.729718954 | proteklo 3.8 s
Newton 4/24 | prihvacen | rho 0.9961 | radijus 5.0000e-01 | Max|grad| 6.842e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.729626525 | proteklo 4.8 s
Newton 5/24 | prihvacen | rho 1.0014 | radijus 5.0000e-01 | Max|grad| 4.617e-03
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.729543186 | proteklo 5.8 s
Newton 6/24 | prihvacen | rho 0.9908 | radijus 5.0000e-01 | Max|grad| 6.848e-03
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.729468208 | proteklo 6.7 s
Newton 7/24 | prihvacen | rho 1.0024 | radijus 5.0000e-01 | Max|grad| 3.938e-03
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.729392311 | proteklo 7.7 s
Newton 8/24 | prihvacen | rho 0.9852 | radijus 5.0000e-01 | Max|grad| 7.267e-03
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.729316478 | proteklo 8.7 s
Newton 9/24 | prihvacen | rho 1.0030 | radijus 5.0000e-01 | Max|grad| 3.936e-03
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.729237161 | proteklo 9.7 s
Newton 10/24 | prihvacen | rho 0.9809 | radijus 5.0000e-01 | Max|grad| 7.594e-03
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.729157211 | proteklo 10.7 s
Newton 11/24 | prihvacen | rho 1.0033 | radijus 5.0000e-01 | Max|grad| 3.825e-03
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.729074005 | proteklo 11.7 s
Newton 12/24 | prihvacen | rho 0.9776 | radijus 5.0000e-01 | Max|grad| 7.827e-03
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.728991794 | proteklo 12.6 s
Newton 13/24 | prihvacen | rho 1.0033 | radijus 5.0000e-01 | Max|grad| 3.636e-03
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.728907530 | proteklo 13.7 s
Newton 14/24 | prihvacen | rho 0.9754 | radijus 5.0000e-01 | Max|grad| 7.973e-03
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.728825671 | proteklo 14.8 s
Newton 15/24 | prihvacen | rho 1.0031 | radijus 5.0000e-01 | Max|grad| 3.467e-03
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.728742246 | proteklo 15.8 s
Newton 16/24 | prihvacen | rho 0.9744 | radijus 5.0000e-01 | Max|grad| 8.180e-03
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.728660610 | proteklo 16.8 s
Newton 17/24 | prihvacen | rho 1.0028 | radijus 5.0000e-01 | Max|grad| 3.512e-03
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.728575611 | proteklo 17.7 s
Newton 18/24 | prihvacen | rho 0.9743 | radijus 5.0000e-01 | Max|grad| 8.691e-03
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.728488895 | proteklo 18.7 s
Newton 19/24 | prihvacen | rho 1.0025 | radijus 5.0000e-01 | Max|grad| 3.743e-03
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.728393624 | proteklo 19.7 s
Newton 20/24 | prihvacen | rho 0.9736 | radijus 5.0000e-01 | Max|grad| 9.736e-03
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.728290284 | proteklo 20.7 s
Newton 21/24 | prihvacen | rho 1.0021 | radijus 5.0000e-01 | Max|grad| 4.204e-03
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.728168559 | proteklo 21.7 s
Newton 22/24 | prihvacen | rho 0.9694 | radijus 5.0000e-01 | Max|grad| 1.149e-02
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.728029317 | proteklo 22.7 s
Newton 23/24 | prihvacen | rho 1.0019 | radijus 5.0000e-01 | Max|grad| 4.919e-03
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.727853879 | proteklo 23.6 s
Newton 24/24 | prihvacen | rho 0.9581 | radijus 5.0000e-01 | Max|grad| 1.418e-02
Najbolji cilj: 16.727646502
Max|gradijent|: 1.418e-02
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

Odvojeni test NLL: 16.701832924
Test ne menja izabranu regularizaciju.

loto7_4712_k87_loto_plus_1734.csv | zavrsni trening cele istorije
Uzorci: 1733 | ridge: 0.10000 | parametri: 36
Proracun 1/120 | Hesijan-vektor 1 | najbolji cilj 16.722855822 | proteklo 0.0 s
Proracun 5/120 | Hesijan-vektor 4 | najbolji cilj 16.722855822 | proteklo 0.9 s
Newton 1/24 | prihvacen | rho 1.0101 | radijus 5.0000e-01 | Max|grad| 1.328e-02
Proracun 10/120 | Hesijan-vektor 8 | najbolji cilj 16.722012673 | proteklo 2.0 s
Newton 2/24 | prihvacen | rho 1.0292 | radijus 5.0000e-01 | Max|grad| 1.068e-02
Proracun 15/120 | Hesijan-vektor 12 | najbolji cilj 16.721592709 | proteklo 3.0 s
Newton 3/24 | prihvacen | rho 1.0056 | radijus 5.0000e-01 | Max|grad| 8.645e-03
Proracun 20/120 | Hesijan-vektor 16 | najbolji cilj 16.721269938 | proteklo 4.0 s
Newton 4/24 | prihvacen | rho 0.9976 | radijus 5.0000e-01 | Max|grad| 9.907e-03
Proracun 25/120 | Hesijan-vektor 20 | najbolji cilj 16.720942597 | proteklo 5.0 s
Newton 5/24 | prihvacen | rho 1.0038 | radijus 5.0000e-01 | Max|grad| 1.030e-02
Proracun 30/120 | Hesijan-vektor 24 | najbolji cilj 16.720580036 | proteklo 6.1 s
Newton 6/24 | prihvacen | rho 0.9912 | radijus 5.0000e-01 | Max|grad| 1.118e-02
Proracun 35/120 | Hesijan-vektor 28 | najbolji cilj 16.720174298 | proteklo 7.2 s
Newton 7/24 | prihvacen | rho 1.0034 | radijus 5.0000e-01 | Max|grad| 1.198e-02
Proracun 40/120 | Hesijan-vektor 32 | najbolji cilj 16.719696962 | proteklo 8.2 s
Newton 8/24 | prihvacen | rho 0.9878 | radijus 5.0000e-01 | Max|grad| 1.345e-02
Proracun 45/120 | Hesijan-vektor 36 | najbolji cilj 16.719150613 | proteklo 9.3 s
Newton 9/24 | prihvacen | rho 1.0016 | radijus 5.0000e-01 | Max|grad| 1.396e-02
Proracun 50/120 | Hesijan-vektor 40 | najbolji cilj 16.718496932 | proteklo 10.3 s
Newton 10/24 | prihvacen | rho 0.9870 | radijus 5.0000e-01 | Max|grad| 1.577e-02
Proracun 55/120 | Hesijan-vektor 44 | najbolji cilj 16.717757140 | proteklo 11.3 s
Newton 11/24 | prihvacen | rho 0.9972 | radijus 5.0000e-01 | Max|grad| 1.613e-02
Proracun 60/120 | Hesijan-vektor 48 | najbolji cilj 16.716888013 | proteklo 12.3 s
Newton 12/24 | prihvacen | rho 0.9909 | radijus 5.0000e-01 | Max|grad| 1.710e-02
Proracun 65/120 | Hesijan-vektor 52 | najbolji cilj 16.715945117 | proteklo 13.3 s
Newton 13/24 | prihvacen | rho 0.9888 | radijus 5.0000e-01 | Max|grad| 1.667e-02
Proracun 70/120 | Hesijan-vektor 56 | najbolji cilj 16.714902576 | proteklo 14.4 s
Newton 14/24 | prihvacen | rho 0.9985 | radijus 5.0000e-01 | Max|grad| 1.623e-02
Proracun 75/120 | Hesijan-vektor 60 | najbolji cilj 16.713840311 | proteklo 15.4 s
Newton 15/24 | prihvacen | rho 0.9778 | radijus 5.0000e-01 | Max|grad| 1.837e-02
Proracun 80/120 | Hesijan-vektor 64 | najbolji cilj 16.712769185 | proteklo 16.4 s
Newton 16/24 | prihvacen | rho 1.0034 | radijus 5.0000e-01 | Max|grad| 1.386e-02
Proracun 85/120 | Hesijan-vektor 68 | najbolji cilj 16.711733719 | proteklo 17.4 s
Newton 17/24 | prihvacen | rho 0.9700 | radijus 5.0000e-01 | Max|grad| 1.834e-02
Proracun 90/120 | Hesijan-vektor 72 | najbolji cilj 16.710770646 | proteklo 18.4 s
Newton 18/24 | prihvacen | rho 1.0043 | radijus 5.0000e-01 | Max|grad| 1.285e-02
Proracun 95/120 | Hesijan-vektor 76 | najbolji cilj 16.709867169 | proteklo 19.5 s
Newton 19/24 | prihvacen | rho 0.9682 | radijus 5.0000e-01 | Max|grad| 1.824e-02
Proracun 100/120 | Hesijan-vektor 80 | najbolji cilj 16.709068209 | proteklo 20.5 s
Newton 20/24 | prihvacen | rho 1.0041 | radijus 5.0000e-01 | Max|grad| 1.138e-02
Proracun 105/120 | Hesijan-vektor 84 | najbolji cilj 16.708332890 | proteklo 21.6 s
Newton 21/24 | prihvacen | rho 0.9713 | radijus 5.0000e-01 | Max|grad| 1.903e-02
Proracun 110/120 | Hesijan-vektor 88 | najbolji cilj 16.707697815 | proteklo 22.6 s
Newton 22/24 | prihvacen | rho 1.0039 | radijus 5.0000e-01 | Max|grad| 9.983e-03
Proracun 115/120 | Hesijan-vektor 92 | najbolji cilj 16.707116787 | proteklo 23.6 s
Newton 23/24 | prihvacen | rho 0.9770 | radijus 5.0000e-01 | Max|grad| 1.813e-02
Proracun 120/120 | Hesijan-vektor 96 | najbolji cilj 16.706612959 | proteklo 24.7 s
Newton 24/24 | prihvacen | rho 1.0039 | radijus 5.0000e-01 | Max|grad| 8.941e-03
Najbolji cilj: 16.706146523
Max|gradijent|: 8.941e-03
Status: Dostignut budzet Newtonovih iteracija.
DIJAGNOSTIKA: stacionarnost nije potvrdjena; zadrzan je najbolji vazeci kandidat.

REZULTAT: loto7_4712_k87_loto_plus_1734.csv
NEXT: 1 3 13 26 32 34 36
Izabran ridge: 0.10000
Validacioni NLL: 16.754636606
Odvojeni test NLL: 16.701832924
Modelna verovatnoca NEXT: 1.16350063341e-06
Zbir svih verovatnoca: 1.000000000000
Greska logaritamske normalizacije: 0.000e+00
Zbir verovatnoca ukljucivanja: 7.000000000000
Broj kombinacija obuhvacenih raspodelom: 15,380,937
Globalni maksimum dobijen je tacnim svojstvom proizvodne raspodele, bez generisanja kombinacija.
Greska Qiskit provere: 6.939e-18
Osetljivost na istoriju: 1.803403583e-01
Zavrsni Max|gradijent|: 8.941e-03
Zavrsni optimizacioni proracuni: 120
Od toga Hesijan-vektor proracuni: 96
Mali gradijent sam po sebi ne dokazuje globalni minimum parametara.
Test histogram pogodaka:
0: 66
1: 146
2: 102
3: 32
4: 1
5: 0
6: 0
7: 0
Prosecan broj pogodaka na testu: 1.296830
Verovatnoce ukljucivanja u NEXT raspodeli:
 1: 0.236408773
 2: 0.146017163
 3: 0.259650010
 4: 0.123878978
 5: 0.168596942
 6: 0.159985434
 7: 0.202655041
 8: 0.158337574
 9: 0.166686869
10: 0.174358353
11: 0.152139482
12: 0.156194601
13: 0.285499411
14: 0.182488091
15: 0.168329565
16: 0.166953415
17: 0.156931230
18: 0.139616150
19: 0.158677278
20: 0.129482413
21: 0.164950598
22: 0.203302528
23: 0.175919980
24: 0.147321341
25: 0.154076860
26: 0.259530298
27: 0.201261778
28: 0.149143854
29: 0.175430481
30: 0.194401869
31: 0.156524953
32: 0.276164359
33: 0.153723600
34: 0.225059991
35: 0.144028140
36: 0.225159147
37: 0.196018650
38: 0.150527549
39: 0.154567252
DIJAGNOSTIKA: Stacionarnost zavrsnog treninga nije potvrdjena.
DIJAGNOSTIKA: Test model ima nepotvrdjenu stacionarnost.
Rezultat nije menjan radi uklanjanja dijagnostickog obrasca.

======================================================================
ZAVRSNI NEXT REZULTATI
loto7_4712_k87.csv | NEXT: 6 x 13 y 27 z 30 | ridge: 0.10000 | test NLL: 16.758623715
loto7_4712_k87_loto_2978.csv | NEXT: 6 x 15 y 29 z 35 | ridge: 0.10000 | test NLL: 16.738173616
loto7_4712_k87_loto_plus_1734.csv | NEXT: 1 x 13 y 32 z 36 | ridge: 0.10000 | test NLL: 16.701832924
"""



"""
Jacobians, Hessians & Newton

učenje kvantnog modela pomoću zakrivljenosti funkcije greške 

- Jakobijan pokazuje kako svaki ugao Qiskit kola menja izlazne amplitude i raspodelu.
- Hesijan pokazuje zakrivljenost NLL-a i međusobnu zavisnost parametara. Može otkriti ravne pravce i nestabilne korake.
- Newtonov postupak sa ograničenjem koraka koristi tu zakrivljenost da predloži promenu uglova. 
Korak prihvatamo prema stvarnom smanjenju funkcije cilja.
Za veći broj parametara računati proizvod Hesijana i vektora, umesto cele matrice. 
Takav pristup podržava SciPy trust-krylov.

V1: Qiskit model raspodele sa učenjem drugog reda, uz proveru izvoda i kontrolu veličine koraka. 
Kvantne amplitude ostaju prediktivni izlaz; Newtonov postupak je klasični optimizator.
V1 sa Newtonovim učenjem drugog reda i kontrolom veličine koraka. 
Qiskit će računati amplitude, a optimizator će koristiti izvode da proceni zakrivljenost NLL-a. 

Provera formula na malom primeru dala je odstupanje oko 5.18e-10 za gradijent i 1.26e-10 za proizvod Hesijana i vektora. 
To je nezavisna provera u JavaScriptu. 
V1 će koristiti Hesijan kroz takve proizvode, bez formiranja cele velike matrice, i sačuvaće najbolji važeći kandidat ako se dostigne budžet računanja.
V1 koristi Newtonov postupak sa oblašću poverenja. 
Jakobijan i proizvod Hesijana i vektora računaju se analitičkim diferenciranjem kvantne evolucije i normalizacije raspodele. 
Trening ne koristi konačne razlike.

Jakobijan, Hesijan i Newtonov metod jesu klasični matematički alati. 
U datom kodu kvantni deo je Qiskit kolo koje proizvodi amplitude. 
Klasični deo koristi njihove izvode za učenje uglova.
Ovo je Qiskit model sa klasičnim učenjem drugog reda; nije kvantni Newtonov algoritam.
"""
