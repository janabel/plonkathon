from utils import *
import py_ecc.bn128 as b
from curve import ec_lincomb, G1Point, G2Point
from compiler.program import CommonPreprocessedInput
from verifier import VerificationKey
from dataclasses import dataclass
from poly import Polynomial, Basis

# Recover the trusted setup from a file in the format used in
# https://github.com/iden3/snarkjs#7-prepare-phase-2
SETUP_FILE_G1_STARTPOS = 80
SETUP_FILE_POWERS_POS = 60


@dataclass
class Setup(object):
    #   ([1]₁, [x]₁, ..., [x^{d-1}]₁)
    # = ( G,    xG,  ...,  x^{d-1}G ), where G is a generator of G_2
    powers_of_x: list[G1Point]
    # [x]₂ = xH, where H is a generator of G_2
    X2: G2Point

    @classmethod
    def from_file(cls, filename):
        contents = open(filename, "rb").read()
        # Byte 60 gives you the base-2 log of how many powers there are
        powers = 2 ** contents[SETUP_FILE_POWERS_POS]
        # Extract G1 points, which start at byte 80
        values = [
            int.from_bytes(contents[i : i + 32], "little")
            for i in range(
                SETUP_FILE_G1_STARTPOS, SETUP_FILE_G1_STARTPOS + 32 * powers * 2, 32
            )
        ]
        assert max(values) < b.field_modulus
        # The points are encoded in a weird encoding, where all x and y points
        # are multiplied by a factor (for montgomery optimization?). We can
        # extract the factor because we know the first point is the generator.
        factor = b.FQ(values[0]) / b.G1[0]
        values = [b.FQ(x) / factor for x in values]
        powers_of_x = [(values[i * 2], values[i * 2 + 1]) for i in range(powers)]
        print("Extracted G1 side, X^1 point: {}".format(powers_of_x[1]))
        # Search for start of G2 points. We again know that the first point is
        # the generator.
        pos = SETUP_FILE_G1_STARTPOS + 32 * powers * 2
        target = (factor * b.G2[0].coeffs[0]).n
        while pos < len(contents):
            v = int.from_bytes(contents[pos : pos + 32], "little")
            if v == target:
                break
            pos += 1
        print("Detected start of G2 side at byte {}".format(pos))
        X2_encoding = contents[pos + 32 * 4 : pos + 32 * 8]
        X2_values = [
            b.FQ(int.from_bytes(X2_encoding[i : i + 32], "little")) / factor
            for i in range(0, 128, 32)
        ]
        X2 = (b.FQ2(X2_values[:2]), b.FQ2(X2_values[2:]))
        assert b.is_on_curve(X2, b.b2)
        print("Extracted G2 side, X^1 point: {}".format(X2))
        # assert b.pairing(b.G2, powers_of_x[1]) == b.pairing(X2, b.G1)
        # print("X^1 points checked consistent")
        return cls(powers_of_x, X2)

    # Encodes the KZG commitment that evaluates to the given values in the group
    def commit(self, values: Polynomial) -> G1Point:
        assert values.basis == Basis.LAGRANGE

        poly = values.fft(inv=True)
        d = len(poly.values)
        # ec_lincomb(pairs) where pairs = list of (pt, coeff), gives you linear combination on elliptic curve 
        pairs = [(self.powers_of_x[i], poly.values[i]) for i in range(d)]
        return ec_lincomb(pairs)
        

    # Generate the verification key for this program with the given setup
    def verification_key(self, pk: CommonPreprocessedInput) -> VerificationKey:
        # Create the appropriate VerificationKey object
        # vp = ([Q], [S]) needs commitment to the selector polynomials (Q) and the permutation polynomials (S)
        vk = VerificationKey(
            group_order=pk.group_order,
            Qm = self.commit(pk.QM),
            Ql = self.commit(pk.QL),
            Qr = self.commit(pk.QR),
            Qo = self.commit(pk.QO),
            Qc = self.commit(pk.QC),
            S1 = self.commit(pk.S1),
            S2 = self.commit(pk.S2),
            S3 = self.commit(pk.S3),
            X_2 = self.X2,
            w = Scalar.root_of_unity(pk.group_order), # root of unity
        )
        
        return vk
    
    # # we set this to some power of 2 (so that we can FFT over it), that is at least the number of constraints we have (so we can Lagrange interpolate them)
    # group_order: int
    # # [q_M(x)]₁ (commitment to multiplication selector polynomial)
    # Qm: G1Point
    # # [q_L(x)]₁ (commitment to left selector polynomial)
    # Ql: G1Point
    # # [q_R(x)]₁ (commitment to right selector polynomial)
    # Qr: G1Point
    # # [q_O(x)]₁ (commitment to output selector polynomial)
    # Qo: G1Point
    # # [q_C(x)]₁ (commitment to constants selector polynomial)
    # Qc: G1Point
    # # [S_σ1(x)]₁ (commitment to the first permutation polynomial S_σ1(X))
    # S1: G1Point
    # # [S_σ2(x)]₁ (commitment to the second permutation polynomial S_σ2(X))
    # S2: G1Point
    # # [S_σ3(x)]₁ (commitment to the third permutation polynomial S_σ3(X))
    # S3: G1Point
    # # [x]₂ = xH, where H is a generator of G_2
    # X_2: G2Point
    # # nth root of unity (i.e. ω^1), where n is the program's group order.
    # w: Scalar
