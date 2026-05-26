//! Narrative demo of the STIR protocol.
//!
//! Run with: `cargo run -p stir --bin demo`
//!
//! The narrative below walks through one execution. As reminders: `L_0`
//! is the initial evaluation domain (an `|L_0|`-element subset of the
//! Goldilocks field `F`), `f_0: L_0 → F` is the prover's input function
//! claimed to be the evaluation table of some polynomial of degree
//! `< d_0` (the initial degree bound), `L_i` is the round-`i` evaluation
//! domain after `i` folding steps, and `α_i` is the round-`i` fold
//! randomness drawn from the Fiat-Shamir transcript.
//!
//! Pedagogy: this demo is meant to be **read** before implementation (to
//! understand the protocol's shape) and **run** after each implementation
//! phase (to see progress). Early on, the call to
//! `run_stir_with_verification` will panic at the first unimplemented
//! function from a sibling module — that's expected, and the demo prints
//! helpful breadcrumbs so you can see exactly which call panicked.
//!
//! The chosen parameters are tiny — `|L_0| = 64`, fold by `4`, two rounds —
//! so the protocol's data shapes fit on a screen. They are NOT
//! cryptographically-meaningful values; `security_bits = 32` is for demo
//! visibility only, NOT production.
//!
//! Compare with `reed_solomon::bin::demo` — that demo had no notion of
//! "verifier", just a single encode/decode pair. STIR adds an entire
//! verifier state machine, so this demo's narrative is more about the
//! per-round handshake than about a single algebraic operation.

use reed_solomon::{Fp, UnivariatePoly};
use stir::params::StirParams;
use stir::protocol::run_stir_with_verification;

fn main() {
    println!("=== STIR low-degree IOP of proximity — narrative demo ===\n");
    println!("All parameters here are DEMO values (security_bits = 32, tiny");
    println!("domain). Production STIR uses security_bits >= 128 and");
    println!("|L_0| >= 2^20. See `coding_plan.md §6` for the full discussion.\n");

    // ════════════════════════════════════════════════════════════════════
    // === Setup ===
    // ════════════════════════════════════════════════════════════════════
    //
    // Demo params, hand-picked so the protocol's data shapes fit on one
    // screen. With fold-by-4 we go d=16 → d=4 → d=1 (a constant) in two
    // rounds; the constant is the final polynomial.
    let log_initial_domain_size: u32 = 6; // |L_0| = 2^6 = 64
    let folding_factor: u32 = 4;
    let num_rounds: u32 = 2;
    let rate_log_inv: u32 = 2; // ρ = 2^{-2} = 1/4
    let security_bits: u32 = 32; // DEMO ONLY — production uses >= 128
    let ood_samples: u32 = 1;
    let stopping_degree: usize = 4;
    let initial_degree_bound: usize = 16; // = |L_0| * ρ = 64 * 1/4
    // NOTE: We do NOT hard-code a repetition schedule here. The schedule
    // is derived from `security_bits` by `StirParams` (per-round soundness
    // `2^{-λ/M}`; for λ = 32, M = 2 this gives ~[8, 6, 4]). Hard-coding
    // `[8, 4, 2]` would under-shoot the bound and deliver only ~8 bits.

    println!("───────────────────────────────────────────────────────────");
    println!("(1) Parameters");
    println!("───────────────────────────────────────────────────────────");
    println!("  log_initial_domain_size = {}  (|L_0| = 2^{} = {})",
        log_initial_domain_size, log_initial_domain_size,
        1u64 << log_initial_domain_size);
    println!("  folding_factor          = {}", folding_factor);
    println!("  num_rounds              = {}", num_rounds);
    println!("  rate_log_inv            = {}  (ρ = 1/{} = {:.3})",
        rate_log_inv, 1 << rate_log_inv, 1.0_f64 / (1 << rate_log_inv) as f64);
    println!("  security_bits           = {}  (DEMO ONLY, not production!)",
        security_bits);
    println!("  ood_samples             = {}", ood_samples);
    println!("  stopping_degree         = {}", stopping_degree);
    println!("  initial_degree_bound    = {}  (= |L_0| · ρ)",
        initial_degree_bound);
    println!();
    println!("  Round-by-round sizing (predicted):");
    println!("    Round 0: |L_0| = 64, d_0 = 16, fold ÷ {}, domain ÷ {}",
        folding_factor, folding_factor / 2);
    println!("    Round 1: |L_1| = 32, d_1 = 4,  fold ÷ {}, domain ÷ {}",
        folding_factor, folding_factor / 2);
    println!("    Final:   |L_2| = 16, d_2 = 1   ← final_polynomial (constant)");
    println!();

    let params = StirParams::new(
        log_initial_domain_size,
        initial_degree_bound,
        folding_factor,
    )
    .with_num_rounds(num_rounds)
    .with_rate_log_inv(rate_log_inv)
    .with_security_bits(security_bits)
    .with_ood_samples(ood_samples)
    .with_stopping_degree(stopping_degree);

    let repetition_schedule = params.repetition_schedule.clone();
    println!("  repetition_schedule     = {:?}  (derived from security_bits)",
        repetition_schedule);

    // ════════════════════════════════════════════════════════════════════
    // === Build a test polynomial ===
    // ════════════════════════════════════════════════════════════════════
    //
    // The witness `f_0` is a degree-15 polynomial (= initial_degree_bound -
    // 1). We just take coefficients [1, 2, ..., 16].
    println!("───────────────────────────────────────────────────────────");
    println!("(2) Test polynomial");
    println!("───────────────────────────────────────────────────────────");

    let coeffs: Vec<Fp> = (1u64..=16).map(Fp::new).collect();
    let polynomial = UnivariatePoly::new(coeffs.clone());
    println!("  f_0(X) = 1 + 2·X + 3·X^2 + ... + 16·X^15");
    println!("  coefficients: {:?}",
        coeffs.iter().map(|c| c.as_u64()).collect::<Vec<_>>());
    println!("  degree: {} (within initial_degree_bound = {})",
        polynomial.degree().map(|d| d as i64).unwrap_or(-1),
        initial_degree_bound);
    println!();

    // ════════════════════════════════════════════════════════════════════
    // === Run STIR ===
    // ════════════════════════════════════════════════════════════════════
    //
    // This calls into prover → verifier. While the implementation is in
    // progress this will PANIC at the first todo!() inside sibling modules
    // (likely transcript, then ood, then fold, ...). That's expected and
    // gives you a clear breadcrumb of which call to implement next.
    println!("───────────────────────────────────────────────────────────");
    println!("(3) Run STIR — prove and verify");
    println!("───────────────────────────────────────────────────────────");
    println!("  Calling `run_stir_with_verification(params, polynomial)`...");
    println!();
    println!("  NOTE: if any sibling module is still stubbed, this will");
    println!("  panic at a `todo!()`. The panic message tells you which");
    println!("  module to implement next.");
    println!();

    let result = run_stir_with_verification(params, polynomial);

    // ════════════════════════════════════════════════════════════════════
    // === Display ===
    // ════════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────");
    println!("(4) Result");
    println!("───────────────────────────────────────────────────────────");

    match result {
        Ok((proof, accepted)) => {
            println!("  run_stir_with_verification returned Ok(...).");
            println!(
                "  Verifier decision: {}",
                if accepted { "ACCEPT" } else { "REJECT" }
            );
            println!();
            println!("  Proof shape:");
            println!("    round_commitments.len()    = {}",
                proof.round_commitments.len());
            println!("    repetition_schedule        = {:?}",
                repetition_schedule);
            println!("    ood_replies.len()          = {}",
                proof.ood_replies.len());
            println!("    shift_answers.len()        = {}",
                proof.shift_answers.len());
            println!("    merkle_paths.len()         = {}",
                proof.merkle_paths.len());
            println!("    final_polynomial.degree()  = {:?}",
                proof.final_polynomial.degree());
            if !proof.pow_nonces.is_empty() {
                println!("    pow_nonces.len()           = {}",
                    proof.pow_nonces.len());
            }
            println!();
            assert!(
                accepted,
                "honest demo run must be accepted by the verifier",
            );
            println!("  verifier accepted");
        }
        Err(msg) => {
            println!("  run_stir_with_verification returned Err({:?}).", msg);
            println!("  This is the structured-error path — the input was");
            println!("  malformed before any cryptography ran. See the");
            println!("  protocol::run_stir docstring for the validation rules.");
            panic!("demo: expected Ok(...), got Err({msg:?})");
        }
    }

    println!();

    // ════════════════════════════════════════════════════════════════════
    // === What you'd see when implemented ===
    // ════════════════════════════════════════════════════════════════════
    println!("───────────────────────────────────────────────────────────");
    println!("(5) Expected narrative (once all sibling modules implement)");
    println!("───────────────────────────────────────────────────────────");
    println!();
    println!("  Round 0 (|L_0| = 64, d_0 = 16):");
    println!("    Prover:   evaluates f_0 on L_0 → 64-element codeword.");
    println!("              Merkle-commits to it. SHA3-256 root sent.");
    println!("              Samples 1 OOD point z_0; replies f_0(z_0).");
    println!("              Samples 8 shift positions; opens 8 Merkle paths.");
    println!("              Draws α_0 ← transcript.");
    println!("              Folds f_0 by 4 → degree-4 polynomial g_0.");
    println!("              Applies OOD quotient (Lemma 4.4) → h_0.");
    println!("              Degree-corrects → f_1 ∈ RS[F, L_1, 4].");
    println!();
    println!("  Round 1 (|L_1| = 32, d_1 = 4):");
    println!("    Same shape with 4 shift queries (per repetition_schedule).");
    println!("    Final fold-by-4 → degree-1 polynomial (a constant or");
    println!("    a degree-0 poly, depending on the witness).");
    println!();
    println!("  Final:");
    println!("    Prover sends final_polynomial (degree < {}) IN THE CLEAR.",
        stopping_degree);
    println!("    Verifier:");
    println!("      - Re-derives all challenges from its own transcript.");
    println!("      - Checks each Merkle path against the round root.");
    println!("      - Checks final_polynomial.degree() < {}.",
        stopping_degree);
    println!("      - Checks consistency between last-round queries and");
    println!("        final_polynomial at the corresponding fold points.");
    println!("    Accepts iff every check passes.");
    println!();
    println!("  The whole protocol is O(log² d_0) verifier queries — for");
    println!("  d_0 = 16 that's about 4² = 16 queries (concretely 8+4 = 12");
    println!("  in this schedule). STIR's win over FRI is that this stays");
    println!("  O(log² d) as d grows, while FRI is O(log d · poly(λ)).");
}
