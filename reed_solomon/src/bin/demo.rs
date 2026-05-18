//! A narrative demo of Reed-Solomon encoding and decoding.
//!
//! Run with: `cargo run -p reed_solomon --bin demo`
//!
//! Pedagogy: prints a small message polynomial, its codeword, simulates one
//! corruption, decodes via interpolation (which won't recover from the
//! corruption — that's the point of Berlekamp-Welch), and re-encodes the
//! recovered message to show the residual error vector.

use reed_solomon::{
    decode::{decode_berlekamp_welch, decode_via_interpolation},
    domain::EvaluationDomain,
    encode::ReedSolomonCode,
    field::Fp,
    polynomial::UnivariatePoly,
};

fn main() {
    // TODO: write a narrative demo. Suggested skeleton:
    //
    //   println!("Reed-Solomon demo over Goldilocks F_p, p = 2^64 - 2^32 + 1\n");
    //
    //   let log_size = 3;            // domain size 8
    //   let degree_bound = 4;        // rate = 4 / 8 = 1/2
    //   let domain = EvaluationDomain::new_subgroup(log_size);
    //   let code = ReedSolomonCode::new(domain.clone(), degree_bound);
    //
    //   let message = UnivariatePoly::new(vec![
    //       Fp::new(1), Fp::new(2), Fp::new(3), Fp::new(4),
    //   ]);
    //   println!("message = {:?}", message);
    //
    //   let codeword = code.encode(&message);
    //   println!("codeword (length {}):", codeword.len());
    //   for (i, c) in codeword.iter().enumerate() {
    //       println!("  c[{}] = {}", i, c.as_u64());
    //   }
    //
    //   let recovered = decode_via_interpolation(&codeword, &domain, degree_bound);
    //   println!("\nrecovered (uncorrupted): {:?}", recovered);
    //   assert_eq!(recovered, message);
    //
    //   // Simulate a single error and observe that interpolation no longer recovers
    //   // the original polynomial — Berlekamp-Welch would, if implemented.
    //   let mut corrupted = codeword.clone();
    //   corrupted[3] = corrupted[3] + Fp::one();  // flip one position
    //   let recovered_bad = decode_via_interpolation(&corrupted, &domain, degree_bound);
    //   println!("\nrecovered (corrupted, no BW): {:?}", recovered_bad);
    //   println!("(this is wrong — interpolation has no notion of error correction)");
    //
    //   // Optional: if you implement Berlekamp-Welch, call it here and show that
    //   // it recovers the original `message` from `corrupted`.

    println!("Reed-Solomon demo over Goldilocks F_p, p = 2^64 - 2^32 + 1");
    println!("================================================================\n");

    // === Setup ===
    let log_size: u32 = 3; // domain size 2^3 = 8
    let degree_bound: usize = 4; // rate = 4 / 8 = 1/2
    // The subgroup case `new_subgroup(log_size)` (offset = 1) is the simpler
    // special case — `fft_on_domain` short-circuits straight to `fft_subgroup`
    // because `c^i = 1` makes the pre-scale a no-op. The coset case
    // (offset != 1) is what every production STARK/FRI pipeline uses for
    // trace-codeword disjointness: see `domain::EvaluationDomain` "Why STARKs
    // use cosets" rustdoc. This demo exercises the coset path so the
    // pre-scale-then-FFT (and de-scale-after-iFFT) branches are actually run.
    let domain = EvaluationDomain::new_coset(log_size, Fp::new(7));
    let code = ReedSolomonCode::new(domain.clone(), degree_bound);

    let (d, n) = code.rate();
    let radius = (n - d) / 2;
    println!("Domain: coset c·⟨ω⟩ of size n = {}, offset c = {}", n, domain.offset().as_u64());
    println!("Degree bound: d = {}", d);
    println!("Rate: ρ = {}/{} = {:.2}", d, n, d as f64 / n as f64);
    println!("Unique-decoding radius: ⌊(n - d) / 2⌋ = {} error(s)\n", radius);

    // === Message polynomial ===
    let message = UnivariatePoly::new(vec![
        Fp::new(1),
        Fp::new(2),
        Fp::new(3),
        Fp::new(4),
    ]);
    println!("Message polynomial: p(X) = 1 + 2·X + 3·X² + 4·X³");
    println!("Coefficient vector: [1, 2, 3, 4]\n");

    // === Encode ===
    let codeword = code.encode(&message);
    println!("Codeword c = (p(c·ω⁰), p(c·ω¹), ..., p(c·ω^{}{})) — length {}:", n - 1, "", codeword.len());
    for (i, c_i) in codeword.iter().enumerate() {
        println!("  c[{:>2}] = {}", i, c_i.as_u64());
    }
    println!();

    // === Decode the clean codeword via interpolation ===
    let recovered = decode_via_interpolation(&codeword, &domain, degree_bound);
    println!("Decoded via inverse FFT (no errors injected):");
    println!("  recovered coefficients: {:?}",
        recovered.coeffs().iter().map(|c| c.as_u64()).collect::<Vec<_>>());
    assert_eq!(recovered, message);
    println!("  ✓ matches original message\n");

    // === Inject one corruption ===
    let error_position = 3;
    let error_delta = Fp::new(42);
    let mut corrupted = codeword.clone();
    corrupted[error_position] = corrupted[error_position] + error_delta;

    println!("Injecting an error: c[{}] += {}", error_position, error_delta.as_u64());
    println!("Corrupted codeword:");
    for (i, c_i) in corrupted.iter().enumerate() {
        let marker = if i == error_position { "  ← ERROR" } else { "" };
        println!("  c[{:>2}] = {}{}", i, c_i.as_u64(), marker);
    }
    println!();

    // === Interpolation-only decoder (no error correction) — gives garbage ===
    let bad = decode_via_interpolation(&corrupted, &domain, degree_bound);
    println!("Decoded via interpolation (no error correction):");
    println!("  recovered coefficients: {:?}",
        bad.coeffs().iter().map(|c| c.as_u64()).collect::<Vec<_>>());
    println!("  matches original? {}", bad == message);
    println!("  ← Interpolation just inverse-FFTs the input; it has no notion");
    println!("    of \"errors\" and faithfully decodes the corrupted vector.\n");

    // === Berlekamp-Welch decoder — recovers the message ===
    match decode_berlekamp_welch(&corrupted, &domain, degree_bound) {
        Ok(p) => {
            println!("Decoded via Berlekamp-Welch (with error correction):");
            println!("  recovered coefficients: {:?}",
                p.coeffs().iter().map(|c| c.as_u64()).collect::<Vec<_>>());
            println!("  matches original? {}", p == message);
            if p == message {
                println!("  ✓ Berlekamp-Welch corrected the 1-position error within radius {}", radius);
            }
        }
        Err(msg) => println!("BW failed: {}", msg),
    }
    println!();

    // === Beyond the unique-decoding radius — BW correctly fails ===
    println!("------------------------------------------------------------");
    println!("Bonus: what happens beyond the unique-decoding radius?");
    println!("------------------------------------------------------------");
    let mut very_corrupted = codeword.clone();
    for i in 0..(radius + 1) {
        very_corrupted[i] = very_corrupted[i] + Fp::new((i + 1) as u64 * 100);
    }
    println!("Injected {} errors (one more than radius {}).", radius + 1, radius);
    match decode_berlekamp_welch(&very_corrupted, &domain, degree_bound) {
        Ok(p) => {
            println!("BW returned a polynomial: {:?}",
                p.coeffs().iter().map(|c| c.as_u64()).collect::<Vec<_>>());
            println!("Matches original? {}", p == message);
            println!("(Note: beyond the radius, BW's output is not guaranteed");
            println!("to be the original message even if it succeeds.)");
        }
        Err(msg) => println!("BW correctly returned Err: \"{}\" ✓", msg),
    }
}
