//! A narrative demo of Reed-Solomon encoding and decoding.
//!
//! Run with: `cargo run -p reed_solomon --bin demo`
//!
//! Pedagogy: prints a small message polynomial, its codeword, simulates one
//! corruption, decodes via interpolation (which won't recover from the
//! corruption — that's the point of Berlekamp-Welch), and re-encodes the
//! recovered message to show the residual error vector.

use reed_solomon::{
    decode::decode_via_interpolation,
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
    let _ = (
        EvaluationDomain::new_subgroup,
        ReedSolomonCode::new,
        UnivariatePoly::new,
        Fp::new,
        decode_via_interpolation,
    );
    todo!()
}
