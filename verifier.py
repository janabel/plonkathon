import py_ecc.bn128 as b
from utils import *
from dataclasses import dataclass
from curve import *
from transcript import Transcript
from poly import Polynomial, Basis

@dataclass
class VerificationKey:
    """Verification key"""

    group_order: int
    # [q_M(x)]₁ (commitment to multiplication selector polynomial)
    Qm: G1Point
    # [q_L(x)]₁ (commitment to left selector polynomial)
    Ql: G1Point
    # [q_R(x)]₁ (commitment to right selector polynomial)
    Qr: G1Point
    # [q_O(x)]₁ (commitment to output selector polynomial)
    Qo: G1Point
    # [q_C(x)]₁ (commitment to constants selector polynomial)
    Qc: G1Point
    # [S_σ1(x)]₁ (commitment to the first permutation polynomial S_σ1(X))
    S1: G1Point
    # [S_σ2(x)]₁ (commitment to the second permutation polynomial S_σ2(X))
    S2: G1Point
    # [S_σ3(x)]₁ (commitment to the third permutation polynomial S_σ3(X))
    S3: G1Point
    # [x]₂ = xH, where H is a generator of G_2
    X_2: G2Point
    # nth root of unity, where n is the program's group order.
    w: Scalar
    # commitment [x]1 (commitment to secret)
    x_1: G1Point

    # More optimized version that tries hard to minimize pairings and
    # elliptic curve multiplications, but at the cost of being harder
    # to understand and mixing together a lot of the computations to
    # efficiently batch them.
    def verify_proof(self, group_order: int, pf, public=[]) -> bool:

        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf) # fiat shamir deterministic, just hashing proof transcript
        # print(f'zeta={zeta}')
        
        proof = pf.flatten()
        a_1 = proof['a_1']
        b_1 = proof['b_1']
        c_1 = proof['c_1']        
        z_1 = proof['z_1']
        t_lo_1 = proof['t_lo_1']
        t_mid_1 = proof['t_mid_1']
        t_hi_1 = proof['t_hi_1']
        W_z_1 = proof['W_z_1']
        W_zw_1 = proof['W_zw_1']
        a_eval = proof['a_eval']
        b_eval = proof['b_eval']
        c_eval = proof['c_eval']
        s1_eval = proof['s1_eval']
        s2_eval = proof['s2_eval']
        z_shifted_eval = proof['z_shifted_eval']
        # print(f'proof = {proof}')

        Qm_1 = self.Qm
        Ql_1 = self.Ql
        Qr_1 = self.Qr
        Qo_1 = self.Qo
        Qc_1 = self.Qc
        S1_1 = self.S1
        S2_1 = self.S2
        S3_1 = self.S3
        X_2 = self.X_2
        w = self.w
        x_1 = self.x_1

        # Verify type checks
        for msg in ['a_1', 'b_1', 'c_1', 'z_1', 't_lo_1', 't_mid_1', 't_hi_1', 'W_z_1', 'W_zw_1']:
            # print(f'proof[msg] = {proof[msg]}, type = {type(proof[msg])}')
            assert type(proof[msg]) == tuple
        print('passed G1Point type checks')

        for msg in ['a_eval', 'b_eval', 'c_eval', 's1_eval', 's2_eval', 'z_shifted_eval']:
            assert type(proof[msg]) == Scalar
        print('passed F_p type checks')

        for input in public:
            assert (type(input) == int or type(input) == Scalar)
        print('passed public (input) type checks')

        # Compute zero poly eval Z_H(zeta) = zeta^n - 1
        ZH_eval = Scalar(zeta ** group_order - 1)
        # print(f'ZH_eval = {ZH_eval}')

        # Compute lagrange polynomial evaluation L1(zeta)
        L1 = Polynomial(values=[Scalar(1)]+[Scalar(0)]*(group_order-1), basis=Basis.LAGRANGE)
        L1_eval = L1.barycentric_eval(zeta)
        # L1_eval = ZH_eval / (group_order * (zeta - 1))
        # print(f'L1_eval from expansion = {L1.barycentric_eval(zeta)}')   
        # print(f'L1_eval ZH_eval / (group_order * (zeta - 1)}')  

        # Compute public input polynomial evaluation PI(zeta)
        # print(f'public inputs = {public}')
        PI = Polynomial(
            [Scalar(-v) for v in public]
            + [Scalar(0) for _ in range(self.group_order - len(public))],
            Basis.LAGRANGE,
        )
        PI_eval = PI.barycentric_eval(zeta)
        # print(f'PI values from verifier = {PI.values}')

        # Compute constant part of r_1 = commit(r)
        r_0 = PI_eval - L1_eval * alpha**2 - alpha * (
            (a_eval + beta * s1_eval + gamma) * (b_eval + beta * s2_eval + gamma) * (c_eval + gamma)
        ) * z_shifted_eval

        # print(f'type r_0 = {type(r_0)}')

        # Compute non constant part of r = r_1
        r_prime_1 = ec_lincomb([
            (Qm_1, a_eval * b_eval),
            (Ql_1, a_eval),
            (Qr_1, b_eval),
            (Qo_1, c_eval),
            (Qc_1, 1),
            (z_1, 
                (a_eval + beta * zeta + gamma)
                * (b_eval + 2 * beta * zeta + gamma) # [TODO] change the 2 to k1 (restructure verifier)
                * (c_eval + 3 * beta * zeta + gamma) * alpha + (L1_eval * alpha**2)),
            (S3_1, -((a_eval + beta * s1_eval + gamma)
                * (b_eval + beta * s2_eval + gamma)
                * (alpha * beta * z_shifted_eval))),
            (t_lo_1, -(ZH_eval)),
            (t_mid_1, -(ZH_eval * zeta**group_order)),
            (t_hi_1, -(ZH_eval * zeta**(2 * group_order))),
            ])

        D_1 = ec_lincomb([(r_prime_1, 1), (z_1, u)])
        # print(f'type(D_1) = {type(D_1)}')

        # Compute full batched polynomial commitment F_1
        F_1 = ec_lincomb([(D_1, 1), (a_1, v), (b_1, v**2), (c_1, v**3), (S1_1, v**4), (S2_1, v**5)])

        # Compute group-encoded batch evaluation (note that separated all constants out to here to reduce # exponentiations)
        E_1 = ec_lincomb([(b.G1, (-r_0 + v * a_eval + v**2 * b_eval + v**3 * c_eval
               + v**4 * s1_eval + v**5 * s2_eval + u * z_shifted_eval))])

        # Batch validate all evaluations
        P1 = ec_lincomb([(W_z_1, 1), (W_zw_1, u)])
        P2 = ec_lincomb([(W_z_1, zeta), (W_zw_1, u * zeta * w), (F_1,1), (E_1,-1)])

        G2 = b.G2
        G1 = b.G1

        # print(f'b_first_pairing = {b.pairing(X_2, P1)}')
        # print(f'b_second_pairing = {b.pairing(G2, P2)}')

        # no way...pairing takes twisted point / curve as first argument
        return b.pairing(X_2, P1) == b.pairing(G2, P2)
    
    # Basic, easier-to-understand version of what's going on.
    # Feel free to use multiple pairings.
    def verify_proof_unoptimized(self, group_order: int, pf, public=[]) -> bool: # this is same as above (optimized) b/c I was lazy

        beta, gamma, alpha, zeta, v, u = self.compute_challenges(pf) # fiat shamir deterministic, just hashing proof transcript
        # print(f'zeta={zeta}')
        
        proof = pf.flatten()
        a_1 = proof['a_1']
        b_1 = proof['b_1']
        c_1 = proof['c_1']        
        z_1 = proof['z_1']
        t_lo_1 = proof['t_lo_1']
        t_mid_1 = proof['t_mid_1']
        t_hi_1 = proof['t_hi_1']
        W_z_1 = proof['W_z_1']
        W_zw_1 = proof['W_zw_1']
        a_eval = proof['a_eval']
        b_eval = proof['b_eval']
        c_eval = proof['c_eval']
        s1_eval = proof['s1_eval']
        s2_eval = proof['s2_eval']
        z_shifted_eval = proof['z_shifted_eval']
        # print(f'proof = {proof}')

        Qm_1 = self.Qm
        Ql_1 = self.Ql
        Qr_1 = self.Qr
        Qo_1 = self.Qo
        Qc_1 = self.Qc
        S1_1 = self.S1
        S2_1 = self.S2
        S3_1 = self.S3
        X_2 = self.X_2
        w = self.w
        x_1 = self.x_1

        # Verify type checks
        for msg in ['a_1', 'b_1', 'c_1', 'z_1', 't_lo_1', 't_mid_1', 't_hi_1', 'W_z_1', 'W_zw_1']:
            # print(f'proof[msg] = {proof[msg]}, type = {type(proof[msg])}')
            assert type(proof[msg]) == tuple
        print('passed G1Point type checks')

        for msg in ['a_eval', 'b_eval', 'c_eval', 's1_eval', 's2_eval', 'z_shifted_eval']:
            assert type(proof[msg]) == Scalar
        print('passed F_p type checks')

        for input in public:
            assert type(input) == int
        print('passed public (input) type checks')

        # Compute zero poly eval Z_H(zeta) = zeta^n - 1
        ZH_eval = Scalar(zeta ** group_order - 1)
        # print(f'ZH_eval = {ZH_eval}')

        # Compute lagrange polynomial evaluation L1(zeta)
        L1 = Polynomial(values=[Scalar(1)]+[Scalar(0)]*(group_order-1), basis=Basis.LAGRANGE)
        L1_eval = L1.barycentric_eval(zeta)
        # L1_eval = ZH_eval / (group_order * (zeta - 1))
        # print(f'L1_eval from expansion = {L1.barycentric_eval(zeta)}')   
        # print(f'L1_eval ZH_eval / (group_order * (zeta - 1)}')  
        

        # Compute public input polynomial evaluation PI(zeta)
        # print(f'public inputs = {public}')
        PI = Polynomial(
            [Scalar(-v) for v in public]
            + [Scalar(0) for _ in range(self.group_order - len(public))],
            Basis.LAGRANGE,
        )
        PI_eval = PI.barycentric_eval(zeta)
        # print(f'PI values from verifier = {PI.values}')

        # Compute constant part of r_1 = commit(r)
        r_0 = PI_eval - L1_eval * alpha**2 - alpha * (
            (a_eval + beta * s1_eval + gamma) * (b_eval + beta * s2_eval + gamma) * (c_eval + gamma)
        ) * z_shifted_eval

        # print(f'type r_0 = {type(r_0)}')

        # Compute non constant part of r = r_1
        r_prime_1 = ec_lincomb([
            (Qm_1, a_eval * b_eval),
            (Ql_1, a_eval),
            (Qr_1, b_eval),
            (Qo_1, c_eval),
            (Qc_1, 1),
            (z_1, 
                (a_eval + beta * zeta + gamma)
                * (b_eval + 2 * beta * zeta + gamma) # [TODO] change the 2 to k1 (restructure verifier)
                * (c_eval + 3 * beta * zeta + gamma) * alpha + (L1_eval * alpha**2)),
            (S3_1, -((a_eval + beta * s1_eval + gamma)
                * (b_eval + beta * s2_eval + gamma)
                * (alpha * beta * z_shifted_eval))),
            (t_lo_1, -(ZH_eval)),
            (t_mid_1, -(ZH_eval * zeta**group_order)),
            (t_hi_1, -(ZH_eval * zeta**(2 * group_order))),
            ])

        # sanity check that r_0 and r_prime_1 computed correctly
        r = ec_lincomb([(r_prime_1, 1), (b.G1, r_0)])
        # print(f'r(zeta), which should be 0 point = point at infinity on elliptic curve, = {r}') # ok yeah this returns None = probs pt at infinity...

        D_1 = ec_lincomb([(r_prime_1, 1), (z_1, u)])
        # print(f'type(D_1) = {type(D_1)}')

        # Compute full batched polynomial commitment F_1
        F_1 = ec_lincomb([(D_1, 1), (a_1, v), (b_1, v**2), (c_1, v**3), (S1_1, v**4), (S2_1, v**5)])

        # Compute group-encoded batch evaluation (note that separated all constants out to here to reduce # exponentiations)
        E_1 = ec_lincomb([(b.G1, (-r_0 + v * a_eval + v**2 * b_eval + v**3 * c_eval
               + v**4 * s1_eval + v**5 * s2_eval + u * z_shifted_eval))])

        # Batch validate all evaluations
        P1 = ec_lincomb([(W_z_1, 1), (W_zw_1, u)])
        P2 = ec_lincomb([(W_z_1, zeta), (W_zw_1, u * zeta * w), (F_1,1), (E_1,-1)])

        G2 = b.G2
        G1 = b.G1

        # print(f'b_first_pairing = {b.pairing(X_2, P1)}')
        # print(f'b_second_pairing = {b.pairing(G2, P2)}')

        # no way...pairing takes twisted point / curve as first argument
        return b.pairing(X_2, P1) == b.pairing(G2, P2)

    # Compute challenges (should be same as those computed by prover)
    def compute_challenges(self, proof
    ) -> tuple[Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
        transcript = Transcript(b"plonk")
        beta, gamma = transcript.round_1(proof.msg_1)
        alpha, _fft_cofactor = transcript.round_2(proof.msg_2)
        zeta = transcript.round_3(proof.msg_3)
        # zeta = int(20000) # for testing verifier with x = zeta = 20000
        v = transcript.round_4(proof.msg_4)
        u = transcript.round_5(proof.msg_5)

        return beta, gamma, alpha, zeta, v, u
