//! The sumcheck verifier, generic over a per-variable summation domain.
//!
//! ## What the verifier does, in one paragraph
//!
//! The verifier sees the public polynomial `g`, the per-variable domain `D`
//! (so `S = D.elements()`, `k = |S|`), and the prover's claim
//! `H = sum_{x in S^n} g(x)`. In each round it receives a `k`-vector
//! `msg = [s_i(elements[0]), ..., s_i(elements[k-1])]`, checks that the
//! sum-over-`S` matches the running expectation, samples a uniformly random
//! `r ∈ F`, and **updates the running expectation to `s_i(r)`** — the value of
//! the univariate slice at the random challenge, obtained by `k`-point
//! Lagrange interpolation. After `n` rounds, it evaluates `g` at the
//! accumulated challenges and compares against the running expectation.
//!
//! ## Two generic invariants
//!
//! Let `S = D.elements()`, `k = |S|`.
//!
//! 1. **Per-round sum-over-`S` check.** The verifier's running expectation
//!    at the start of round `i` is `current_claim` (initially `claimed_sum`).
//!    The round-`i` message `msg` should satisfy
//!
//!    ```text
//!    sum over j in 0..k of msg[j] == current_claim.
//!    ```
//!
//!    *Why:* `msg[j] = s_i(elements[j])`, and `s_i` was defined by the prover
//!    as the partial sum of `g` along all but its first variable. Plugging in
//!    the definition,
//!
//!    ```text
//!    sum_{j ∈ [0, k)} s_i(elements[j])
//!      = sum_{x_1 ∈ S}  sum_{(x_2, ..., x_{n-i+1}) ∈ S^{n-i}} g(..., x_1, ..., x_{n-i+1})
//!      = current_claim.
//!    ```
//!
//!    For `D = BooleanHypercube` (`k = 2`) this is just `msg[0] + msg[1] ==
//!    current_claim`, the familiar two-add Boolean check. No special case
//!    needed — it's the `k = 2` slice of the same general formula.
//!
//! 2. **`s_i(r)` from `k`-point Lagrange interpolation.** The verifier samples
//!    `r ∈ F` uniformly and updates `current_claim = s_i(r)`. By the **Lagrange
//!    interpolation theorem**, the unique polynomial of degree `< k` through
//!    the `k` known points `(elements[j], msg[j])` is
//!
//!    ```text
//!    s_i(X) = sum over j in 0..k of L_j(X) · msg[j],
//!    ```
//!
//!    where `L_j` is the **Lagrange basis polynomial** with
//!    `L_j(elements[j]) = 1` and `L_j(elements[i]) = 0` for `i ≠ j`. Evaluated
//!    at `X = r`,
//!
//!    ```text
//!    s_i(r) = sum over j in 0..k of L_j(r) · msg[j].
//!    ```
//!
//!    For `D = BooleanHypercube` (`k = 2`, `elements = [0, 1]`),
//!    `L_0(r) = 1 - r` and `L_1(r) = r`, so the formula collapses to
//!    `(1 - r) · msg[0] + r · msg[1] = msg[0] + r · (msg[1] - msg[0])` —
//!    the familiar line interpolation from the original Boolean-only code.
//!    Again no special case — it's the `k = 2` slice of the same formula.
//!
//! See [`lagrange_basis_at`] for the basis-evaluator implementation; it is a
//! duplicate of the private helper in [`crate::polynomial`] kept private to
//! this module on purpose (we don't want to add it to the polynomial-layer
//! public API).
//!
//! **Question for the reader.** Suppose a cheating prover sends a polynomial
//! `s̃_i(X)` differing from the honest `s_i(X)` — i.e., the polynomial
//! `s̃_i − s_i` is a nonzero polynomial of degree `< k`. What's the chance a
//! uniform-random `r ∈ F` is a coincident point? Why does that bound
//! translate, via the union bound, to overall soundness `n(k-1)/|F|`?
//! Try to answer before reading on.
//!
//! `s̃_i - s_i` is a *nonzero* polynomial of degree `< k`, so by the polynomial-
//! root bound it has at most `k - 1` roots in `F`. A coincident point is a
//! root of this difference, so `Pr[uniform r ∈ F is coincident] ≤ (k-1)/|F|`.
//! The union bound across the `n` rounds (each round samples its own
//! independent `r_i`) sums the per-round failure probabilities, giving
//! `n · (k-1) / |F|` overall.
//!
//! ## Soundness (per-round and total)
//!
//! Per-round, a cheating prover commits to a `s_i(X)` it claims to be the
//! honest slice. By the **polynomial-root bound** (a nonzero polynomial of
//! degree `< k` has at most `k - 1` roots in `F`), if the prover lied, the
//! cheating polynomial agrees with the honest one at most `k - 1` field
//! points — a uniformly random `r ∈ F` lands on one of those with probability
//! at most `(k - 1) / |F|`. So:
//!
//! ```text
//! Pr[verifier fooled in a single round] ≤ (|D| - 1) / |F|.
//! ```
//!
//! Union-bound across the `n` rounds:
//!
//! ```text
//! Pr[verifier fooled overall] ≤ n · (|D| - 1) / |F|.
//! ```
//!
//! Concretely on this crate's `F_p` (`p = 2^61 - 1`):
//!
//! - `|D| = 2` (BooleanHypercube): `≤ n / 2^61`. For `n = 20`, `~2^-56.5`.
//! - `|D| = 3` (Interval3):       `≤ 2n / 2^61`. For `n = 20`, `~2^-55.7`.
//!
//! Both well past cryptographic comfort. The point of `Interval3` here is to
//! exercise the generic Lagrange path, not to weaken soundness meaningfully.
//!
//! ## Why the polynomial-root bound holds (Schwartz-Zippel for univariate)
//!
//! The soundness argument above leans on this fact: a nonzero univariate
//! `p(X) ∈ F[X]` of degree `< k` has at most `k - 1` roots in `F`.
//!
//! Proof by induction on degree (factor theorem + degree drop).
//!
//! - Base `deg p = 0`. Then `p` is a nonzero constant, so it has **zero** roots
//!   — and `0 ≤ k - 1` since `k ≥ 1`.
//! - Inductive step. Suppose true for all degrees `< d`, and let `deg p = d`
//!   with `d ≥ 1`. If `p` has no roots we are done (`0 ≤ k - 1`). Otherwise
//!   pick a root `α ∈ F`. Set `q(X) = p(X) / (X - α)`. By the factor theorem
//!   (since `α` is a root of `p`), `q ∈ F[X]` has degree exactly `d - 1`. By
//!   the induction hypothesis (applied at degree `d - 1 < d`), `q` has at most
//!   `d - 1` roots in `F`. Every root `β ≠ α` of `p` satisfies `q(β) = 0`
//!   (since `F` has no zero divisors), so the other roots of `p` are exactly
//!   the roots of `q`. Counting: at most `d - 1` such roots, plus `α` itself,
//!   gives at most `1 + (d - 1) = d` roots for `p`. Since `d ≤ k - 1`, we are
//!   done. Degree can't drop below 0, so the recursion terminates. ∎
//!
//! Applied to the cheating-prover slice `s̃_i(X) - s_i(X)` (degree `< k`,
//! nonzero by assumption): at most `k - 1` roots in `F`, so a uniform random
//! `r` lands on a root with probability `≤ (k - 1) / |F|`. Across `n` rounds,
//! the union bound gives total soundness error `≤ n · (k - 1) / |F|`.
//!
//! ## Why the verifier check is `sum_j msg[j] == prev_claim` exactly
//!
//! It's the *defining* consistency condition. The prover commits to
//! `msg = [s_i(elements[0]), ..., s_i(elements[k-1])]` where
//! `s_i(X) = sum over rest of g(..., X, ..., rest)`. By definition
//!
//! ```text
//! sum_{j ∈ [0, k)} msg[j] = sum_{j} s_i(elements[j])
//!                         = sum_{j} sum_{rest} g(elements[j], rest)
//!                         = sum_{(x_1, rest) ∈ S × S^{n_remaining - 1}} g(...)
//!                         = sum over S^{n_remaining} of g.
//! ```
//!
//! The previous round's verifier-side update guarantees that this last
//! quantity is what `prev_claim` should equal (either it's the original
//! `claimed_sum`, or it's `s_{i-1}(r_{i-1})` which the prior round's Lagrange
//! interpolation already committed to). Any mismatch means the prover lied
//! somewhere — the verifier rejects.
//!
//! ## Final check
//!
//! After all `n` rounds, `current_claim` is the verifier's expectation for
//! `g(r_1, ..., r_n)`. [`final_check`](SumcheckVerifier::final_check)
//! evaluates `g` directly at the accumulated challenges (this is the **one**
//! place the verifier touches `g`) and compares. The supplied polynomial must
//! use the **same** [`SumDomain`] as the verifier was constructed with;
//! [`SumcheckVerifier<D, R>`] enforces this at the type level.

use crate::domain::SumDomain;
use crate::field::Fp;
use crate::polynomial::MultivariatePoly;
use rand::Rng;

/// Sumcheck verifier state, generic over a per-variable summation domain `D`
/// and an RNG `R`.
pub struct SumcheckVerifier<D: SumDomain, R: Rng> {
    /// The per-variable summation domain; defines `k = |D|` and the
    /// interpolation nodes `elements()`.
    domain: D,
    /// The prover's original claim about `H = sum over D^n_vars of g`.
    /// Stored for inspection; the live running expectation is `current_claim`.
    #[allow(dead_code)]
    claimed_sum: Fp,
    /// Total number of variables `n` — equivalently, the number of rounds the
    /// protocol will run.
    n_vars: usize,
    /// Running expectation, updated each round:
    /// - Before round 0: equals `claimed_sum`.
    /// - After round `i`: equals `s_i(r_{i+1})`.
    current_claim: Fp,
    /// Challenges sampled so far. Length grows from 0 to `n_vars`.
    challenges: Vec<Fp>,
    /// RNG used to sample challenges.
    rng: R,
}

impl<D: SumDomain, R: Rng> SumcheckVerifier<D, R> {
    /// Construct a verifier from the summation domain, the prover's claimed
    /// sum, the number of variables, and an RNG.
    ///
    /// The verifier expects to receive exactly `n_vars` round messages, each
    /// of length `|domain|`, before [`final_check`](Self::final_check) is
    /// called.
    pub fn new(domain: D, claimed_sum: Fp, n_vars: usize, rng: R) -> Self {
        Self {
            domain,
            claimed_sum,
            n_vars,
            current_claim: claimed_sum,
            challenges: vec![],
            rng,
        }
    }

    /// Process round-`i` message
    /// `msg = [s_i(elements[0]), ..., s_i(elements[k-1])]`.
    ///
    /// Three checks, then a state update:
    /// 1. **Length:** `msg.len() == |D|`.
    /// 2. **Sum-over-`S`:** `sum_j msg[j] == current_claim` (see module docs).
    /// 3. **Lagrange interp:** sample `r ∈ F`, set
    ///    `current_claim = sum_j L_j(r) · msg[j]` via [`lagrange_basis_at`]
    ///    (Lagrange interpolation theorem).
    ///
    /// On success returns the sampled `r` so the orchestrator can forward it
    /// to the prover. On failure returns a static error string and leaves the
    /// verifier state otherwise unchanged.
    ///
    /// For `D = BooleanHypercube` this is exactly the Boolean-only
    /// `(1 - r) · msg[0] + r · msg[1]` line update: check
    /// `msg[0] + msg[1] == current_claim`, then set
    /// `current_claim = (1 - r) * msg[0] + r * msg[1]`.
    pub fn process_round_message(&mut self, msg: Vec<Fp>) -> Result<Fp, &'static str> {
        // TODO: perform the round-`i` verifier work in this order.
        //   1. Assert `msg.len() == |D|` (a length-`k` evaluation vector).
        //   2. Sum-over-`S` check: `msg.iter().sum() == self.current_claim`.
        //      That is the definition of `s_i` summed over its `k`-element domain.
        //   3. Sample a fresh `r ∈ F` uniformly via `self.rng` — this is the
        //      verifier's only source of soundness (see polynomial-root bound below).
        //   4. Lagrange-interpolate `s_i(r) = sum_j L_j(r) * msg[j]` through the
        //      `k` known points `(elements[j], msg[j])`.
        //   5. Push `r` to `self.challenges` and set `self.current_claim = s_i(r)`,
        //      then return `Ok(r)` so the orchestrator can forward `r` to the prover.
        //   See the "Soundness (per-round and total)" and "Schwartz-Zippel"
        //   sections above for why a uniform `r` makes cheating succeed with
        //   probability `≤ (k - 1) / |F|` per round.
        //
        //   Reference implementation below.

        let elements = self.domain.elements();
        let k = elements.len();
        if msg.len() != k {
            return Err("round message length does not match |D|");
        }

        // (1) Sum-over-S check. For k = 2 this is `msg[0] + msg[1]`.
        let sum: Fp = msg.iter().copied().sum();
        if sum != self.current_claim {
            return Err("round message does not sum to current claim");
        }

        // (2) Sample r and Lagrange-interpolate s_i at r through the k known
        // points (elements[j], msg[j]). For k = 2 this collapses to the
        // familiar line (1 - r) * msg[0] + r * msg[1]; see module docs.
        let r = Fp::random(&mut self.rng);
        let s_at_r: Fp = (0..k)
            .map(|j| lagrange_basis_at(elements, j, r) * msg[j])
            .sum();

        // (3) Commit the round update.
        self.current_claim = s_at_r;
        self.challenges.push(r);
        Ok(r)
    }

    /// Final consistency check: `g(r_1, ..., r_n_vars) == self.current_claim`.
    ///
    /// `g` must be a multivariate polynomial on the **same** [`SumDomain`]
    /// this verifier was constructed with; the type signature enforces this
    /// (both use the same `D`). This is the one place the verifier evaluates
    /// `g` directly — every prior round used only what the prover sent.
    pub fn final_check(&self, g: &MultivariatePoly<D>) -> Result<(), &'static str> {
        // TODO: verify the final reduction by evaluating `g` at the accumulated challenges.
        //   1. Assert all `n_vars` rounds have been processed — otherwise the
        //      reduction chain is incomplete and `current_claim` isn't yet
        //      supposed to equal `g(r_1, ..., r_n)`.
        //   2. Compute `g(r_1, ..., r_n)` directly. This is the one place the
        //      verifier evaluates `g`; every prior round used only what the
        //      prover sent.
        //   3. Accept iff `g(r_1, ..., r_n) == current_claim`.
        // See the module-doc "Soundness" derivation for why the chain ends here.
        //
        // Reference implementation below.

        assert_eq!(
            self.challenges.len(),
            self.n_vars,
            "SumcheckVerifier::final_check: must process all rounds first",
        );
        if g.evaluate(&self.challenges) == self.current_claim {
            Ok(())
        } else {
            Err("SumcheckVerifier::final_check: g(r_1, ..., r_n) != current_claim")
        }
    }

    /// The challenges sampled so far. Useful for tests and the demo.
    pub fn challenges(&self) -> &[Fp] {
        &self.challenges
    }
}

/// Evaluate the Lagrange basis polynomial `L_i` at `target`, where the basis
/// is built over `points`:
///
/// ```text
/// L_i(target) = prod_{j ≠ i} (target - points[j]) / (points[i] - points[j]).
/// ```
///
/// `L_i` is the unique degree-`< points.len()` polynomial that is `1` at
/// `points[i]` and `0` at every other `points[j]`.
///
/// **Note:** This is a deliberate **duplicate** of the same-named private
/// helper in [`crate::polynomial`]. Both layers (polynomial-fold and verifier-
/// interp) need it, and we keep it private to each module rather than exposing
/// it through the polynomial-layer public API. Signature kept identical.
///
/// The caller must ensure that the elements of `points` are pairwise distinct;
/// otherwise the denominator `(points[i] - points[j])` for some `j ≠ i` would
/// be zero and have no field inverse. In this crate `points` is always
/// [`SumDomain::elements`], which is required to be distinct by the
/// [`SumDomain`] contract.
fn lagrange_basis_at(points: &[Fp], i: usize, target: Fp) -> Fp {
    let mut numerator = Fp::one();
    let mut denominator = Fp::one();
    let xi = points[i];
    for (j, &xj) in points.iter().enumerate() {
        if j == i {
            continue;
        }
        numerator *= target - xj;
        denominator *= xi - xj;
    }
    // SAFE: denominator is a product of nonzero terms because `points` is
    // pairwise distinct (SumDomain contract / caller's responsibility).
    numerator
        * denominator
            .inverse()
            .expect("Lagrange denominator is zero — domain elements must be distinct")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::BooleanHypercube;
    use rand::SeedableRng;

    #[test]
    fn first_round_check_passes_on_consistent_message() {
        // Verifier with claimed_sum = 10 on the Boolean cube (k = 2).
        // msg = [3, 7] has sum 10, matching current_claim, so the check passes.
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut verifier = SumcheckVerifier::new(BooleanHypercube, Fp::new(10), 2, rng);
        let msg = vec![Fp::new(3), Fp::new(7)];
        assert!(verifier.process_round_message(msg).is_ok());
    }

    #[test]
    fn first_round_check_fails_on_inconsistent_message() {
        // msg = [3, 5] has sum 8 ≠ 10, so the verifier rejects.
        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut verifier = SumcheckVerifier::new(BooleanHypercube, Fp::new(10), 2, rng);
        let msg = vec![Fp::new(3), Fp::new(5)];
        assert!(verifier.process_round_message(msg).is_err());
    }
}
