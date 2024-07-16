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

    def flatten(self): # oh makes sense to have flatten here and keep separate from rest of code
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
        # print(f'witness = {witness}')

        # Collect fixed and public information
        # FIXME: Hash pk and PI into transcript
        public_vars = self.program.get_public_assignments()
        PI = Polynomial(
            [Scalar(-witness[v]) for v in public_vars]
            + [Scalar(0) for _ in range(self.group_order - len(public_vars))],
            Basis.LAGRANGE,
        )
        self.PI = PI
        # print(f'PI values from prover = {PI.values}')

        # Round 1
        msg_1 = self.round_1(witness)
        self.beta, self.gamma = transcript.round_1(msg_1) # hash transcript to get permutation challenges beta, gamma

        # Round 2
        msg_2 = self.round_2()
        self.alpha, self.fft_cofactor = transcript.round_2(msg_2) # same thing here. fft_cofactor is random shift

        # Round 3
        msg_3 = self.round_3()
        self.zeta = transcript.round_3(msg_3)

        # for testing intermediate values in verifier. 
        # if make linearization challenge zeta same as secret testing pt x for polynomials, then should be able to check r(zeta)=0 directly.
        # print(f'after round 3, made self.zeta = 20000')
        # self.zeta = int(20000) 
        # assert self.setup.powers_of_x[1] == ec_lincomb([(b.G1, self.zeta)])
        # print(f'zeta and powers of x correct')

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

        # print(f'witness = {witness}')
        # print(f'constraints = {self.program.constraints}')
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

        # print('passed round 1 sanity check')

        a_1 = self.setup.commit(values=self.A)
        b_1 = self.setup.commit(values=self.B)
        c_1 = self.setup.commit(values=self.C)

        return Message1(a_1, b_1, c_1)

    def round_2(self) -> Message2:
        group_order = self.group_order
        n = group_order
        setup = self.setup

        # jk they just set k1=2, k2=3, column 1 is still 1
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
        # print(f'Z_values = {Z_values}')

        # Check that the last term Z_n = 1
        assert Z_values.pop() == 1
        # print('passed last Z value check')

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

        # print('passed round 2 sanity check')

        self.Z = Polynomial(values=Z_values, basis=Basis.LAGRANGE)
        z_1 = self.setup.commit(values=self.Z)

        self.check_polynomial_degree(self.Z, n-1)
        # print('passed Z degree = n-1 check')

        # Return z_1 (commitment to Z polynomial)
        return Message2(z_1)

    def round_3(self) -> Message3:
        group_order = self.group_order
        fft_cofactor = self.fft_cofactor
        setup = self.setup

        # Expand L0 into the coset extended Lagrange basis
        L0_big = self.fft_expand(
            Polynomial([Scalar(1)] + [Scalar(0)] * (group_order - 1), Basis.LAGRANGE)
        )

        # compute q roots of unity, where q = w**(1/4) where w is nth root of unity (so 4nth root of unity)
        self.roots_of_unity_q = Scalar.roots_of_unity(4*group_order)
        
        # self.ZH = Polynomial(values=[Scalar(0)] * self.group_order, basis=Basis.LAGRANGE)
        # ^^ oh wait this doesn't work b/c it's a degree n poly, but lagrange thinks it's just 0 poly (bc restricts to deg n-1 polys)
        # ZH(X) = X^n-1 where n is group order
        # have to compute ZH_big manually. [TODO] FORGOT ABOUT OFFSETS HERE ? 
        roots_of_unity_4 = Scalar.roots_of_unity(4) # ZH(X) = X^n - 1
        # print(f'roots_of_unity_4 = {roots_of_unity_4}')
        ZH_big_vals = [Scalar(self.fft_cofactor**group_order * roots_of_unity_4[i % 4]-1) for i in range(4*group_order)] # X^n - 1 for all X in {q^0, q^1, ..., q^{4n-1}}, where q**4 = omega
        ZH_big = Polynomial(values=ZH_big_vals, basis=Basis.LAGRANGE)
        self.ZH_big = ZH_big
        # ZH_big_coeffs = self.expanded_evals_to_coeffs(ZH_big) # to take inverse, make sure evaluating at offset*q^0, offset*q^1, ... 
        # print(f'ZH_big_coeffs.values = {ZH_big_coeffs.values}') # ZH is correct now, degree n
        
        A_big = self.fft_expand(self.A)
        B_big = self.fft_expand(self.B)
        C_big = self.fft_expand(self.C)
        QL_big = self.fft_expand(self.pk.QL)
        QR_big = self.fft_expand(self.pk.QR)
        QO_big = self.fft_expand(self.pk.QO)
        QM_big = self.fft_expand(self.pk.QM)
        QC_big = self.fft_expand(self.pk.QC)
        PI_big = self.fft_expand(self.PI)
        S1_big = self.fft_expand(self.pk.S1)
        S2_big = self.fft_expand(self.pk.S2)
        S3_big = self.fft_expand(self.pk.S3)
        Z_big = self.fft_expand(self.Z)


        # manually compute evaluation of full quotient poly t on offset*q^0, offset*q^1, ...
        t_values = [Scalar(0) for i in range(group_order * 4)]
        for i in range(group_order * 4):
            t_values[i] = (A_big.values[i] * QL_big.values[i] \
                        + B_big.values[i] * QR_big.values[i] \
                        + C_big.values[i] * QO_big.values[i] \
                        + A_big.values[i] * B_big.values[i] * QM_big.values[i] \
                        + PI_big.values[i] \
                        + QC_big.values[i]) / ZH_big.values[i] \
                        + (
                            (self.rlc(A_big.values[i], self.roots_of_unity_q[i] * self.fft_cofactor)
                            * self.rlc(B_big.values[i], self.k1 * self.roots_of_unity_q[i] * self.fft_cofactor)
                            * self.rlc(C_big.values[i], self.k2 * self.roots_of_unity_q[i] * self.fft_cofactor)
                            ) * Z_big.values[i]
                            - (self.rlc(A_big.values[i], S1_big.values[i])
                            * self.rlc(B_big.values[i], S2_big.values[i])
                            * self.rlc(C_big.values[i], S3_big.values[i])
                            ) * Z_big.values[(i + 4) % (group_order * 4)] # z(Xw) is 4 places over from z(X) in new lagrangian form (4n pts)
                        ) * self.alpha / ZH_big.values[i] \
                        + (
                            (Z_big.values[i] - 1) * L0_big.values[i]
                        ) * self.alpha**2 / ZH_big.values[i]
            
        # print(f't values = {t_values}')
            
        # degree of our quotient poly ~ 2n which is lower than 4n, so coeffs -> lagrange = list[4n] should still maintain all info :)
        QUOT_big = Polynomial(values=t_values, basis=Basis.LAGRANGE) # get lagrangian form of t
        # print(f'list of t evaluations: {QUOT_big.values}')

        # expanded evals to coeffs
        # coset_extended_lagrange_to_coeffs(self, offset): # convert from evaluations of f(offset*x) on x=[q^0, q^1, ...] to coeffs representing poly f(x)
        # Sanity check: QUOT has degree < 3n
        QUOT_big_coeffs = self.expanded_evals_to_coeffs(QUOT_big).values
        # print(f'QUOT_big_coeffs = {QUOT_big_coeffs}, length = {len(QUOT_big_coeffs)}')
        assert (
            QUOT_big_coeffs[-group_order:]
            == [0] * group_order
        )
        # print("Generated the quotient polynomial")

        # computing T1, T2, T3 such that t(X) = T1(X) + X^n * T2(X) + X^2n * T3(X)
        T1_coeffs = QUOT_big_coeffs[:group_order]
        T2_coeffs = QUOT_big_coeffs[group_order:2*group_order]
        T3_coeffs = QUOT_big_coeffs[2*group_order:3*group_order]
        T1 = Polynomial(values=T1_coeffs, basis=Basis.MONOMIAL).fft()
        T2 = Polynomial(values=T2_coeffs, basis=Basis.MONOMIAL).fft()
        T3 = Polynomial(values=T3_coeffs, basis=Basis.MONOMIAL).fft()
        self.T1 = T1
        self.T2 = T2
        self.T3 = T3

        # Sanity check that we've computed T1, T2, T3 correctly
        assert (
            T1.barycentric_eval(fft_cofactor)
            + T2.barycentric_eval(fft_cofactor) * fft_cofactor**group_order
            + T3.barycentric_eval(fft_cofactor) * fft_cofactor ** (group_order * 2)
        ) == QUOT_big.values[0]

        # print('passed round 3 sanity check')

        # no random blinding factors yet, just commit to initial T1, T2, T3

        # print("Generated T1, T2, T3 polynomials")

        self.t_lo_1 = setup.commit(values=T1)
        self.t_mid_1 = setup.commit(values=T2)
        self.t_hi_1 = setup.commit(values=T3)
        
        # Return t_lo_1, t_mid_1, t_hi_1
        return Message3(self.t_lo_1, self.t_mid_1, self.t_hi_1)

    def round_4(self) -> Message4:

        # print(f'BEGINNING ROUND 4 TESTS')
        # print(f'================')

        S1 = self.pk.S1
        S2 = self.pk.S2
        S3 = self.pk.S3
        
        zeta = self.zeta
        print(f'zeta={zeta}')

        self.a_eval = self.A.barycentric_eval(zeta)
        self.b_eval = self.B.barycentric_eval(zeta)
        self.c_eval = self.C.barycentric_eval(zeta)
        self.s1_eval = S1.barycentric_eval(zeta)
        self.s2_eval = S2.barycentric_eval(zeta)

        self.z_shifted_eval = self.Z.barycentric_eval(zeta*self.w)

        # Return a_eval, b_eval, c_eval, s1_eval, s2_eval, z_shifted_eval
        return Message4(self.a_eval, self.b_eval, self.c_eval, self.s1_eval, self.s2_eval, self.z_shifted_eval)

    def round_5(self) -> Message5:

        group_order = self.group_order
        v = self.v
        zeta = self.zeta
        roots_of_unity = self.roots_of_unity

        QM = self.pk.QM
        QL = self.pk.QL
        QR = self.pk.QR
        QO = self.pk.QO
        QC = self.pk.QC
        S1 = self.pk.S1
        S2 = self.pk.S2
        S3 = self.pk.S3

        L1_vals = [Scalar(1)] + [Scalar(0)] * (group_order-1)
        L1 = Polynomial(values=L1_vals, basis=Basis.LAGRANGE)
        self.L1_eval = L1.barycentric_eval(self.zeta)

        # compute linearisation polynomial - ultimately want to check this is 0 at random challenge zeta.
        # linear combinations of the separate checks.
        # notice that we are linear in the polynomials now -> control degree to be < n, so lagrange representation in original basis possible
        # (note: for computing polynomials - as long as can control degree, computing in lagrangian basis is much easier. turns into element-wise operations)
        r_values = [Scalar(0) for i in range(group_order)]
        for i in range(group_order):
            r_values[i] = (self.a_eval * self.b_eval * QM.values[i]
                           + self.a_eval * QL.values[i]
                           + self.b_eval * QR.values[i]
                           + self.c_eval * QO.values[i]
                           + self.PI.barycentric_eval(zeta)
                           + QC.values[i]
                            ) + self.alpha * (
                                self.rlc(self.a_eval, zeta)
                                * self.rlc(self.b_eval, self.k1 * zeta)
                                * self.rlc(self.c_eval, self.k2 * zeta)
                                * self.Z.values[i]
                                - 
                                self.rlc(self.a_eval, self.s1_eval)
                                * self.rlc(self.b_eval, self.s2_eval)
                                * self.rlc(self.c_eval, S3.values[i])
                                * self.z_shifted_eval
                            ) + self.alpha**2 * (
                                (self.Z.values[i] - 1) * self.L1_eval
                            ) - (zeta**group_order - 1) * (
                                self.T1.values[i] + self.T2.values[i] * zeta**group_order + self.T3.values[i] * zeta**(2*group_order)
                            )
            
        R = Polynomial(values=r_values, basis=Basis.LAGRANGE)
        self.R = R

        # Sanity-check R
        assert self.R.barycentric_eval(self.zeta) == 0

        # print("Generated linearization polynomial R")

        # Check that degree of R is not greater than n
        R_expanded = self.fft_expand(self.R)
        R_coeffs = self.expanded_evals_to_coeffs(R_expanded).values
        # print(f'R_coeffs = {R_coeffs}')
        assert R_coeffs[group_order:] == [0] * (group_order * 3)

        # print("Degree of R < n")

        # Construct opening proof polynomial W_z
        W_z_vals = [Scalar(0) for i in range(group_order)]
        for i in range(group_order):
            W_z_vals[i] = 1 / (roots_of_unity[i] - zeta) * (
                            R.values[i]
                            + v * (self.A.values[i] - self.a_eval)
                            + v**2 * (self.B.values[i] - self.b_eval)
                            + v**3 * (self.C.values[i] - self.c_eval)
                            + v**4 * (self.pk.S1.values[i] - self.s1_eval)
                            + v**5 * (self.pk.S2.values[i] - self.s2_eval)
            )
        self.W_z = Polynomial(values=W_z_vals, basis=Basis.LAGRANGE)

        # Check that degree of W_z is not greater than n
        W_z_expanded = self.fft_expand(self.W_z)
        W_z_coeffs = self.expanded_evals_to_coeffs(W_z_expanded).values
        assert W_z_coeffs[group_order:] == [0] * (group_order * 3)

        # print(f'Degree of W_z < n')

        # Construct opening proof polynomial W_zw
        W_zw_vals = [Scalar(0) for i in range(group_order)]
        for i in range(group_order):
            W_zw_vals[i] = (self.Z.values[i] - self.z_shifted_eval) / (roots_of_unity[i] - zeta*self.w)
        self.W_zw = Polynomial(values=W_zw_vals, basis=Basis.LAGRANGE)

        # Check that degree of W_z is not greater than n
        W_zw_expanded = self.fft_expand(self.W_zw)
        W_zw_coeffs = self.expanded_evals_to_coeffs(W_zw_expanded).values
        assert W_zw_coeffs[group_order:] == [0] * (group_order * 3)

        # print(f'Degree of W_zw < n')

        # Compute W_z_1 commitment to W_z
        self.W_z_1 = self.setup.commit(self.W_z)
        # Compute W_zw_1 commitment to W_zw
        self.W_zw_1 = self.setup.commit(self.W_zw)

        # print("Generated final quotient witness polynomials")

        # Return W_z_1, W_zw_1
        return Message5(self.W_z_1, self.W_zw_1)

    def fft_expand(self, x: Polynomial):
        return x.to_coset_extended_lagrange(self.fft_cofactor)

    def expanded_evals_to_coeffs(self, x: Polynomial):
        return x.coset_extended_lagrange_to_coeffs(self.fft_cofactor)

    def rlc(self, term_1, term_2):
        return term_1 + term_2 * self.beta + self.gamma

    def check_polynomial_degree(self, poly, deg): # test if a polynomial is of degree deg
        coeff_poly = poly
        if poly.basis == Basis.LAGRANGE:
            coeff_poly = poly.ifft()
        assert coeff_poly.values[deg+1:] == [Scalar(0)] * len(coeff_poly.values[deg+1:])