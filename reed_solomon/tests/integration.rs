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

/// Pipeline closure: FFT-encode then iFFT-decode on a smooth subgroup recovers the original message.
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
    let domain = EvaluationDomain::new_subgroup(4); // size 16
    let code = ReedSolomonCode::new(domain.clone(), 4);

    let mut rng = rng();
    let coeffs: Vec<Fp> = (0..4).map(|_| Fp::random(&mut rng)).collect();
    let message = UnivariatePoly::new(coeffs);

    let codeword = code.encode(&message);
    let recovered = decode_via_interpolation(&codeword, &domain, 4);

    assert_eq!(recovered, message);
}

/// Same round-trip on a proper coset — exercises the pre-scale / de-scale change-of-variables.
#[test]
fn encode_decode_round_trip_coset() {
    // TODO: same as above but using new_coset with a non-1 offset (e.g., Fp::new(7)).
    let domain = EvaluationDomain::new_coset(4, Fp::new(7)); // size 16, offset 7
    let code = ReedSolomonCode::new(domain.clone(), 4);

    let mut rng = rng();
    let coeffs: Vec<Fp> = (0..4).map(|_| Fp::random(&mut rng)).collect();
    let message = UnivariatePoly::new(coeffs);

    let codeword = code.encode(&message);
    let recovered = decode_via_interpolation(&codeword, &domain, 4);

    assert_eq!(recovered, message);
}

/// Independent oracle: the fast FFT encoder must agree with naive Horner evaluation on every domain point.
#[test]
fn encode_naive_matches_fft_on_random_inputs() {
    // TODO: For multiple random message polynomials, encode() and encode_naive()
    // must agree exactly. Try a few different (log_size, degree_bound) pairs:
    //   (3, 2), (3, 4), (4, 3), (4, 8), (5, 6).
    let mut rng = rng();
    let cases: &[(u32, usize)] = &[(3, 2), (3, 4), (4, 3), (4, 8), (5, 6)];

    for &(log_size, degree_bound) in cases {
        let domain = EvaluationDomain::new_subgroup(log_size);
        let code = ReedSolomonCode::new(domain, degree_bound);

        let coeffs: Vec<Fp> = (0..degree_bound).map(|_| Fp::random(&mut rng)).collect();
        let message = UnivariatePoly::new(coeffs);

        let fast = code.encode(&message);
        let slow = code.encode_naive(&message);

        assert_eq!(
            fast, slow,
            "FFT and naive encode disagree at (log_size={}, degree_bound={})",
            log_size, degree_bound
        );
    }
}

/// Rate `ρ = d / n` accessor reports the unreduced fraction faithfully.
#[test]
fn rate_property_holds() {
    // TODO: code = ReedSolomonCode with degree_bound=2, log_size=3 → size 8.
    // assert code.rate() == (2, 8).
    let domain = EvaluationDomain::new_subgroup(3); // size 8
    let code = ReedSolomonCode::new(domain, 2);

    assert_eq!(code.rate(), (2, 8));
}

/// Codeword length invariant: every encoded codeword has length `|L| = domain.size()`.
#[test]
fn codeword_length_matches_domain() {
    // TODO: codeword.len() == domain.size() for any code.
    let domain = EvaluationDomain::new_subgroup(4); // size 16
    let code = ReedSolomonCode::new(domain.clone(), 4);

    let message = UnivariatePoly::new(vec![
        Fp::new(1),
        Fp::new(2),
        Fp::new(3),
        Fp::new(4),
    ]);
    let codeword = code.encode(&message);

    assert_eq!(codeword.len(), domain.size());
}

/// Smoke test for the encode-decode pipeline on the constant polynomial.
#[test]
fn decoder_recovers_constant_polynomial() {
    // TODO: encode a constant polynomial p(X) = 42; codeword should be [42; n].
    // Decode; recovered polynomial should be the constant 42.
    let domain = EvaluationDomain::new_subgroup(3); // size 8
    let n = domain.size();
    let code = ReedSolomonCode::new(domain.clone(), 4);

    let c = Fp::new(42);
    let message = UnivariatePoly::new(vec![c]); // p(X) = 42

    let codeword = code.encode(&message);
    assert_eq!(codeword, vec![c; n]);

    let recovered = decode_via_interpolation(&codeword, &domain, 4);
    assert_eq!(recovered, message);
}
