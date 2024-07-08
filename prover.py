from compiler.program import Program, CommonPreprocessedInput
from utils import *
from setup import *
from typing import Optional
from dataclasses import dataclass
from transcript import Transcript, Message1, Message2, Message3, Message4, Message5
from poly import Polynomial, Basis
import random


@dataclass
class Proof:
    msg_1: Message1
    msg_2: Message2
    msg_3: Message3
    msg_4: Message4
    msg_5: Message5

    def flatten(self):
        proof = {}
        proof["a_1"] = self.msg_1.a_1
        proof["b_1"] = self.msg_1.b_1
        proof["c_1"] = self.msg_1.c_1
        proof["z_1"] = self.msg_2.z_1
        proof["t_lo_1"] = self.msg_3.t_lo_1
        proof["t_mid_1"] = self.msg_3.t_mid_1
        proof["t_hi_1"] = self.msg_3.t_hi_1
        proof["a_eval"] = self.msg_4.a_eval
        proof["b_eval"] = self.msg_4.b_eval
        proof["c_eval"] = self.msg_4.c_eval
        proof["s1_eval"] = self.msg_4.s1_eval
        proof["s2_eval"] = self.msg_4.s2_eval
        proof["z_shifted_eval"] = self.msg_4.z_shifted_eval
        proof["W_z_1"] = self.msg_5.W_z_1
        proof["W_zw_1"] = self.msg_5.W_zw_1
        return proof


@dataclass
class Prover:
    group_order: int
    setup: Setup
    program: Program
    pk: CommonPreprocessedInput

    def __init__(self, setup: Setup, program: Program):
        self.group_order = program.group_order
        self.setup = setup
        self.program = program
        self.pk = program.common_preprocessed_input()
        # compute and store root of unity
        self.w = self.setup.verification_key(self.pk).w
        self.roots_of_unity = Scalar.roots_of_unity(self.group_order)

    def prove(self, witness: dict[Optional[str], int]) -> Proof:
        # Initialise Fiat-Shamir transcript
        transcript = Transcript(b"plonk")
        self.W = witness

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1) # hash transcript to get permutation challenges beta, gamma

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2) # same thing here

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # Round 4
        msg_4 = self.round_4()
        self.v = transcript.round_4(msg_4)

        # Round 5
        msg_5 = self.round_5()

        return Proof(msg_1, msg_2, msg_3, msg_4, msg_5)

    def round_1(
        self,
        witness: dict[Optional[str], int],
    ) -> Message1:
        program = self.program
        setup = self.setup
        group_order = self.group_order
        constraints = program.constraints
        n = group_order # just for ease

        print(f'witness = {witness}')
        print(f'constraints = {self.program.constraints}')
        self.program.fill_variable_assignments(witness)

        if None not in witness:
            witness[None] = 0

        A = [Scalar(0) for _ in range(self.group_order)]
        B = [Scalar(0) for _ in range(self.group_order)]
        C = [Scalar(0) for _ in range(self.group_order)]
        
        for i, constraint in enumerate(self.program.constraints): # woah new way to get access to indices and values at the same time, remember this
            A[i] = Scalar(witness[constraint.wires.L])
            B[i] = Scalar(witness[constraint.wires.R])
            C[i] = Scalar(witness[constraint.wires.O])

        self.computation_trace = {'a': A, 'b': B, 'c': C}
        
        self.A = Polynomial(values=A, basis=Basis.LAGRANGE)
        self.B = Polynomial(values=B, basis=Basis.LAGRANGE)
        self.C = Polynomial(values=C, basis=Basis.LAGRANGE)
        
        # Sanity check that witness fulfills gate constraints
        assert (
            self.A * self.pk.QL
            + self.B * self.pk.QR
            + self.A * self.B * self.pk.QM
            + self.C * self.pk.QO
            + self.PI
            + self.pk.QC
            == Polynomial([Scalar(0)] * group_order, Basis.LAGRANGE)
        )

        print('passed round 1 sanity check')

        a_1 = self.setup.commit(values=self.A)
        b_1 = self.setup.commit(values=self.B)
        c_1 = self.setup.commit(values=self.C)

        return Message1(a_1, b_1, c_1)

    def round_2(self) -> Message2:
        group_order = self.group_order
        n = group_order
        setup = self.setup

        # already have self.beta, self.gamma
        # # need to compute k1, k2 such that H = {w^i}_i, k1*H, and k2*H are distinct cosets of F
        # roots_of_unity = Scalar.roots_of_unity(n)
        # field_elts = [Scalar(i) for i in range(n)]
        # k1_choices = list(set(field_elts)-set(roots_of_unity))
        # self.k1 = random.sample(k1_choices, 1)

        # k1_coset = [Scalar(k1*root) for root in roots_of_unity]
        # k2_choices = list((set(field_elts)-set(roots_of_unity))-set(k1_coset))
        # self.k2 = random.sample(k2_choices, 1)

        # jk they just set k1=2, k2=3, column 1 is still 1 (?)
        self.k1 = 2
        self.k2 = 3

        # calculate lagrangian representation of the second half of z(X)
        z_coeffs = [Scalar(0)] * n
        z_coeffs[0] = 1

        S1 = self.pk.S1
        S2 = self.pk.S2
        S3 = self.pk.S3

        # L_i evaluates to 1 at omega^i.
        # Z_vals = [Z(omega), Z(omega^2), ..., Z(omega^n)=Z(1)]
        # want Z(w^{n+1}) = Z(omega) = 1 (last elt)

        def z_coeff(i): 
            product = 1
            for j in range(i):
                f_prime_a = self.A.values[j] + self.beta * self.roots_of_unity[j] + self.gamma
                f_prime_b = self.B.values[j] + self.beta * self.k1 * self.roots_of_unity[j] + self.gamma
                f_prime_c = self.C.values[j] + self.beta * self.k2 * self.roots_of_unity[j] + self.gamma
                g_prime_a = self.A.values[j] + self.beta * S1.values[j] + self.gamma
                g_prime_b = self.B.values[j] + self.beta * S2.values[j] + self.gamma
                g_prime_c = self.C.values[j] + self.beta * S3.values[j] + self.gamma
                product *= (f_prime_a * f_prime_b * f_prime_c) / (g_prime_a * g_prime_b * g_prime_c)
            return Scalar(product)
                
        Z_values = [z_coeff(i) for i in range(self.group_order + 1)] # evaluate 0 (Z(omega^0) = 1) to n running product, should have Z(omega^n)=Z(omega^0)=1
        print(f'Z_values = {Z_values}')

        # Check that the last term Z_n = 1
        assert Z_values.pop() == 1
        print('passed last Z value check')

        # def rlc(self, term_1, term_2): oops coulda used this
        #   return term_1 + term_2 * self.beta + self.gamma

        # Sanity-check that Z was computed correctly
        for i in range(group_order):
            assert (
                self.rlc(self.A.values[i], self.roots_of_unity[i])
                * self.rlc(self.B.values[i], 2 * self.roots_of_unity[i])
                * self.rlc(self.C.values[i], 3 * self.roots_of_unity[i])
            ) * Z_values[i] - (
                self.rlc(self.A.values[i], self.pk.S1.values[i])
                * self.rlc(self.B.values[i], self.pk.S2.values[i])
                * self.rlc(self.C.values[i], self.pk.S3.values[i])
            ) * Z_values[
                (i + 1) % group_order
            ] == 0

        print('passed round 2 sanity check')

        self.Z = Polynomial(values=Z_values, basis=Basis.LAGRANGE)
        z_1 = self.setup.commit(values=self.Z)

        # Return z_1 (commitment to Z polynomial)
        return Message2(z_1)

    def round_3(self) -> Message3:
        group_order = self.group_order
        setup = self.setup

        # Expand L0 into the coset extended Lagrange basis
        L0_big = self.fft_expand(
            Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        )



        # Sanity check: QUOT has degree < 3n
        assert (
            self.expanded_evals_to_coeffs(QUOT_big).values[-group_order:]
            == [0] * group_order
        )
        print("Generated the quotient polynomial")

        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            T1.barycentric_eval(fft_cofactor)
            + T2.barycentric_eval(fft_cofactor) * fft_cofactor**group_order
            + T3.barycentric_eval(fft_cofactor) * fft_cofactor ** (group_order * 2)
        ) == QUOT_big.values[0]

        print('passed round 3 sanity check')

        print("Generated T1, T2, T3 polynomials")

        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(t_lo_1, t_mid_1, t_hi_1)

    def round_4(self) -> Message4:

        # Return a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval)

    def round_5(self) -> Message5:

        # Sanity-check R
        assert R.barycentric_eval(zeta) == 0

        print("Generated linearization polynomial R")

        # Check that degree of W_z is not greater than n
        assert W_z_coeffs[group_order:] == [0] * (group_order * 3)

        # Check that degree of W_z is not greater than n
        assert W_zw_coeffs[group_order:] == [0] * (group_order * 3)

        # Compute W_z_1 commitment to W_z

        print("Generated final quotient witness polynomials")

        # Return W_z_1, W_zw_1
        return Message5(W_z_1, W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma
