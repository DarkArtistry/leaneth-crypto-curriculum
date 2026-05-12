//! End-to-end tests for Reed-Solomon encoding and decoding.
//!
//! Where `src/<module>.rs` unit tests check individual functions, this file
//! exercises the **full pipeline**: build a code, encode, decode, compare.
//! These are the tests that catch wiring bugs (e.g., the encoder uses the
//! domain generator but the decoder uses its inverse).

use rand::SeedableRng;
use reed_solomon::{
    decode::decode_via_interpolation,
    domain::EvaluationDomain,
    encode::ReedSolomonCode,
    field::Fp,
    polynomial::UnivariatePoly,
};

/// Build a fresh seeded RNG. Use a fixed seed across tests for reproducibility.
fn rng() -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(0x_5EED_5EED_5EED_5EED)
}

#[test]
fn encode_decode_round_trip_subgroup() {
    // TODO:
    //   1. domain = EvaluationDomain::new_subgroup(4)   (size 16)
    //   2. code = ReedSolomonCode::new(domain.clone(), 4)   (degree bound 4 → rate 1/4)
    //   3. message = UnivariatePoly with 4 random Fp coefficients (use rng() for randomness)
    //   4. codeword = code.encode(&message)
    //   5. recovered = decode_via_interpolation(&codeword, &domain, 4)
    //   6. assert_eq!(recovered, message)
    //
    // Note step 1 panics on overlap — only ONE EvaluationDomain instance can
    // be moved into ReedSolomonCode::new and then queried elsewhere. Hence the
    // `.clone()` in step 2 (or alternatively, build two domains).
    todo!()
}

#[test]
fn encode_decode_round_trip_coset() {
    // TODO: same as above but using new_coset with a non-1 offset (e.g., Fp::new(7)).
    todo!()
}

#[test]
fn encode_naive_matches_fft_on_random_inputs() {
    // TODO: For multiple random message polynomials, encode() and encode_naive()
    // must agree exactly. Try a few different (log_size, degree_bound) pairs:
    //   (3, 2), (3, 4), (4, 3), (4, 8), (5, 6).
    todo!()
}

#[test]
fn rate_property_holds() {
    // TODO: code = ReedSolomonCode with degree_bound=2, log_size=3 → size 8.
    // assert code.rate() == (2, 8).
    todo!()
}

#[test]
fn codeword_length_matches_domain() {
    // TODO: codeword.len() == domain.size() for any code.
    todo!()
}

#[test]
fn decoder_recovers_constant_polynomial() {
    // TODO: encode a constant polynomial p(X) = 42; codeword should be [42; n].
    // Decode; recovered polynomial should be the constant 42.
    todo!()
}
