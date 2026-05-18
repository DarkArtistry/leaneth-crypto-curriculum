//! A narrative demo of the sumcheck protocol — the Boolean-hypercube case.
//!
//! Run with: `cargo run --bin demo`
//!
//! This crate's `run_sumcheck` is generic over the per-variable summation
//! domain `D: SumDomain`. The demo instantiates that generic protocol on
//! `D = BooleanHypercube` (`S = {0, 1}`) — the case Lean Ethereum and every
//! modern multilinear SNARK use.
//!
//! **All field-op counts are MEASURED at runtime** via
//! [`sumcheck::field::op_counter`] — per-thread counters that increment on
//! every `Fp` `+`, `-`, `*`, and `Neg`. No formula-predicted numbers: every
//! count is the program's actual work, snapshotted before and after each
//! phase via `op_counter::snapshot()`.

use rand::SeedableRng;
use sumcheck::domain::BooleanHypercube;
use sumcheck::field::{op_counter, Fp};
use sumcheck::polynomial::MultivariatePoly;
use sumcheck::prover::SumcheckProver;
use sumcheck::verifier::SumcheckVerifier;

fn main() {
    // === Setup ===
    let n: usize = 10; // 2^10 = 1024 hypercube points — small but exponential enough to see the trade-offs.
    let mut rng = rand::rngs::StdRng::seed_from_u64(42); // seeded for reproducibility.

    let evals: Vec<Fp> = (0..(1u64 << n)).map(|_| Fp::random(&mut rng)).collect();
    let poly = MultivariatePoly::new(BooleanHypercube, n, evals);

    println!("=== Sumcheck protocol demo (Boolean-hypercube instance) ===\n");
    println!(
        "This crate's sumcheck protocol is generic over any finite summation\n\
         domain S (see `sumcheck::domain::SumDomain`). The Boolean hypercube\n\
         S = {{0, 1}} instantiated below is the case Lean Ethereum and every\n\
         modern multilinear SNARK uses. The same `run_sumcheck` works\n\
         unchanged for S = Interval3 ({{0, 1, 2}}), as shown by the\n\
         integration tests.\n"
    );
    println!("All field-op counts below are MEASURED via `op_counter::snapshot()`");
    println!("around each phase — every number is the program's actual work,");
    println!("not a formula prediction.\n");

    println!("Polynomial:");
    println!("  type:       multilinear g: F_p^n → F_p");
    println!("  variables:  n = {}", n);
    println!("  summation:  S = {{0, 1}}, k = |S| = 2");
    println!("  total pts:  k^n = 2^n = {}", 1u64 << n);
    println!();

    // ────────────────────────────────────────────────────────────────────
    // (1) Naive sum — MEASURED.
    //
    // `sum_over_hypercube` is `evals.iter().copied().sum()` — pure adds via
    // `<Fp as Sum>::sum`. We expect ~2^n adds (the fold from `Fp::zero()`
    // touches every element). Snapshot before, run, snapshot after, diff.
    // ────────────────────────────────────────────────────────────────────
    let before = op_counter::snapshot();
    let naive_sum = poly.sum_over_hypercube();
    let naive_ops = op_counter::snapshot().since(&before);

    println!("───────────────────────────────────────────────────────────");
    println!("(1) Naive sum");
    println!("───────────────────────────────────────────────────────────");
    println!("  Method: sum the 2^n precomputed MLE evaluations.");
    println!(
        "  Cost (MEASURED): {} adds, {} subs, {} muls, {} negs",
        naive_ops.adds, naive_ops.subs, naive_ops.muls, naive_ops.negs
    );
    println!("                   → {} field ops total", naive_ops.total());
    println!("  Result: H = {}", naive_sum.as_u64());
    println!();

    // ────────────────────────────────────────────────────────────────────
    // (2) Sumcheck protocol — every phase MEASURED.
    //
    // Per round we take three snapshots:
    //   - around the prover's `compute_round_message`
    //   - around the verifier's `process_round_message`
    //   - around the prover's `receive_challenge`
    //
    // Then one final snapshot around the verifier's `final_check`, which
    // performs the one and only g-evaluation the protocol asks of the
    // verifier.
    // ────────────────────────────────────────────────────────────────────
    println!("───────────────────────────────────────────────────────────");
    println!("(2) Sumcheck protocol");
    println!("───────────────────────────────────────────────────────────");

    let claim_sum = poly.sum_over_hypercube();
    let mut prover = SumcheckProver::new(poly.clone());
    let mut verifier = SumcheckVerifier::new(BooleanHypercube, claim_sum, n, rng);

    let mut prover_total: u64 = 0;
    let mut verifier_rounds_total: u64 = 0;

    println!(
        "  Prover commits to claimed sum: H = {}  (matches naive ✓)",
        claim_sum.as_u64()
    );
    println!();

    for i in 0..n {
        // Prover: build the round message.
        let before = op_counter::snapshot();
        let msg = prover.compute_round_message();
        let prover_msg_ops = op_counter::snapshot().since(&before);

        // Snapshot s_i(0), s_i(1) before the verifier consumes `msg`.
        let s_i_0 = msg[0];
        let s_i_1 = msg[1];

        // Verifier: sum-over-S check + |D|-point Lagrange interp at r.
        let before = op_counter::snapshot();
        let r = verifier.process_round_message(msg).unwrap();
        let verifier_round_ops = op_counter::snapshot().since(&before);

        // Prover: fix_first_variable(r) — fold the polynomial.
        let before = op_counter::snapshot();
        prover.receive_challenge(r);
        let prover_update_ops = op_counter::snapshot().since(&before);

        let prover_round_total = prover_msg_ops.total() + prover_update_ops.total();
        prover_total += prover_round_total;
        verifier_rounds_total += verifier_round_ops.total();

        // Narrate first 3 + last round only.
        let narrate = i < 3 || i == n - 1;
        if narrate {
            println!(
                "  Round {:>2}: s_{}(0)={:<18} s_{}(1)={:<18} verifier: {:>4} ops  prover: {:>5} ops",
                i + 1,
                i + 1,
                s_i_0.as_u64(),
                i + 1,
                s_i_1.as_u64(),
                verifier_round_ops.total(),
                prover_round_total
            );
        } else if i == 3 {
            println!("  ... ({} more rounds elided) ...", n - 4);
        }
    }

    // Final check: the verifier's one and only g-evaluation.
    let before = op_counter::snapshot();
    let result = verifier.final_check(&poly);
    let final_ops = op_counter::snapshot().since(&before);

    let verifier_total = verifier_rounds_total + final_ops.total();

    println!();
    println!("  Final check: verifier calls g.evaluate(r_1, ..., r_n) ONCE.");
    println!(
        "  Result: {}",
        if result.is_ok() { "ACCEPT ✓" } else { "REJECT ✗" }
    );
    println!();
    println!("  MEASURED totals:");
    println!(
        "    Verifier round work ({} rounds):  {:>6} field ops",
        n, verifier_rounds_total
    );
    println!(
        "    Verifier final g.evaluate:        {:>6} field ops",
        final_ops.total()
    );
    println!(
        "    Verifier grand total:             {:>6} field ops",
        verifier_total
    );
    println!(
        "    Prover total:                     {:>6} field ops",
        prover_total
    );
    println!();

    // ────────────────────────────────────────────────────────────────────
    // (3) Comparison — what the measurements actually say.
    // ────────────────────────────────────────────────────────────────────
    println!("───────────────────────────────────────────────────────────");
    println!("(3) Comparison");
    println!("───────────────────────────────────────────────────────────");
    println!();
    println!("                                         MEASURED ops");
    println!("    Naive (sum MLE-eval table):          {:>6}", naive_ops.total());
    println!("    Sumcheck verifier (rounds + final):  {:>6}", verifier_total);
    println!();
    println!("  In this toy, naive looks cheap: `sum_over_hypercube` exploits");
    println!(
        "  the precomputed MLE table and is just {} adds. Sumcheck's verifier",
        naive_ops.total()
    );
    println!("  work is dominated by two implementation costs orthogonal to");
    println!("  the protocol itself:");
    println!();
    println!("    1. Each round: 2 Lagrange-basis calls, each computing one");
    println!("       Fermat inverse via `pow(p-2)` — about 121 muls apiece.");
    println!(
        "       That's why each round measures ~{} ops instead of the",
        verifier_rounds_total / n as u64
    );
    println!("       \"4 ops\" the formula would predict. A production");
    println!("       verifier caches the per-domain denominators (constant");
    println!("       for a fixed S), dropping per-round cost to a handful.");
    println!();
    println!("    2. Final g.evaluate: this crate's `MultivariatePoly` stores");
    println!("       g as 2^n eval-table entries and recovers g(r_1,...,r_n)");
    println!(
        "       by repeated `fix_first_variable` — itself O(2^n) ({} ops",
        final_ops.total()
    );
    println!("       measured here). Production sumcheck stores g as a");
    println!("       *structured circuit* where one evaluation is O(|circuit|),");
    println!("       not O(2^n).");
    println!();
    println!("  The structural claim — verifier evaluates g exactly ONCE vs");
    println!("  naive's 2^n times — is unaffected by either knob. With both");
    println!("  fixes (cached domain inverses, structured g), sumcheck's");
    println!("  verifier work is O(n), versus naive's O(2^n × |circuit|).");
    println!();
    println!(
        "  Note: the *prover* shoulders {} ops in this run. Sumcheck is a",
        prover_total
    );
    println!("  delegation protocol — the prover bears the exponential cost so");
    println!("  the verifier only has to check O(n) round messages plus one");
    println!("  g.evaluate. In real SNARKs the verifier is the gas-constrained");
    println!("  or resource-limited party, so its asymptotic win is the");
    println!("  practical win even when the toy's constants mask it here.");
}
