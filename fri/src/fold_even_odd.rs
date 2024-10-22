use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_slice_index_bits};
use tracing::instrument;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_interpolation::interpolate_coset;

/// Fold a polynomial
/// ```ignore
/// p(x) = p_even(x^2) + x p_odd(x^2)
/// ```
/// into
/// ```ignore
/// p_even(x) + beta p_odd(x)
/// ```
/// Expects input to be bit-reversed evaluations.
#[instrument(skip_all, level = "debug")]
pub fn fold_even_odd<F: TwoAdicField>(poly: Vec<F>, beta: F) -> Vec<F> {
    // We use the fact that
    //     p_e(x^2) = (p(x) + p(-x)) / 2
    //     p_o(x^2) = (p(x) - p(-x)) / (2 x)
    // that is,
    //     p_e(g^(2i)) = (p(g^i) + p(g^(n/2 + i))) / 2
    //     p_o(g^(2i)) = (p(g^i) - p(g^(n/2 + i))) / (2 g^i)
    // so
    //     result(g^(2i)) = p_e(g^(2i)) + beta p_o(g^(2i))
    //                    = (1/2 + beta/2 g_inv^i) p(g^i)
    //                    + (1/2 - beta/2 g_inv^i) p(g^(n/2 + i))
    let m = RowMajorMatrix::new(poly, 2);
    let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + 1).inverse();
    let one_half = F::two().inverse();
    let half_beta = beta * one_half;

    // TODO: vectorize this (after we have packed extension fields)

    // beta/2 times successive powers of g_inv
    let mut powers = g_inv
        .shifted_powers(half_beta)
        .take(m.height())
        .collect_vec();
    reverse_slice_index_bits(&mut powers);

    m.par_rows()
        .zip(powers)
        .map(|(mut row, power)| {
            let (r0, r1) = row.next_tuple().unwrap();
            (one_half + power) * r0 + (one_half - power) * r1
        })
        .collect()
}

pub fn fold_poly<F: TwoAdicField>(poly: Vec<F>, beta: F, folding_factor: usize) -> Vec<F> {
    assert!(poly.len() % folding_factor == 0, "The length of the poly must be divisible by the folding factor");
    
    let m = RowMajorMatrix::new(poly, folding_factor);
    let dft = Radix2Dit::default();
    
    let beta_powers: Vec<F> = (0..folding_factor).map(|i| beta.exp_u64(i as u64)).collect();
    // Parallel processing and caching beta powers
    m.row_slices().map(|row| {
        let mut row_evals = row.to_vec();
        let mut row_coeff = dft.idft(row_evals);

        row_coeff.iter().enumerate().fold(F::zero(), |acc, (power, coeff)| {
            acc + (*coeff * beta_powers[power])
        })
    }).collect::<Vec<F>>()
}

pub fn yu_fold_poly<F: TwoAdicField>(poly: Vec<F>, beta: F, folding_factor: usize) -> Vec<F> {
    let log_folding_factor  = log2_ceil_usize(folding_factor);
    
    let mut folded_poly = poly;
    let betas = (1..log_folding_factor+1).into_iter().map(|power|{
        beta.exp_u64(power as u64)
    }).collect::<Vec<F>>();
    for i in 0..log_folding_factor{
        folded_poly = fold_even_odd(folded_poly, betas[i])
    }
    folded_poly
}


#[cfg(test)]
mod tests {
    use itertools::izip;
    use p3_baby_bear::BabyBear;
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
    use rand::{thread_rng, Rng};

    use super::*;

    #[test]
    fn test_fold_even_odd() {
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 10;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let even_coeffs = coeffs.iter().cloned().step_by(2).collect_vec();
        let even_evals = dft.dft(even_coeffs);

        let odd_coeffs = coeffs.iter().cloned().skip(1).step_by(2).collect_vec();
        let odd_evals = dft.dft(odd_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(even_evals, odd_evals)
            .map(|(even, odd)| even + beta * odd)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        reverse_slice_index_bits(&mut folded);
        folded = fold_even_odd(folded, beta);
        reverse_slice_index_bits(&mut folded);

        assert_eq!(expected, folded);
    }

    #[test]
    fn test_fold_poly(){
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 3;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let even_coeffs = coeffs.iter().cloned().step_by(2).collect_vec();
        let even_evals = dft.dft(even_coeffs);

        let odd_coeffs = coeffs.iter().cloned().skip(1).step_by(2).collect_vec();
        let odd_evals = dft.dft(odd_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(even_evals, odd_evals)
            .map(|(even, odd)| even + beta * odd)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        reverse_slice_index_bits(&mut folded);
        folded = yu_fold_poly(folded, beta,2);
        reverse_slice_index_bits(&mut folded);

        assert_eq!(expected, folded);   
    }

    #[test]
    fn test_fold_poly_1(){
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 3;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let p_0_coeffs = coeffs.iter().cloned().step_by(4).collect_vec();
        let p_0_evals = dft.dft(p_0_coeffs);

        let p_1_coeffs = coeffs.iter().cloned().skip(1).step_by(4).collect_vec();
        let p_1_evals = dft.dft(p_1_coeffs);

        let p_2_coeffs = coeffs.iter().cloned().skip(2).step_by(4).collect_vec();
        let p_2_evals = dft.dft(p_2_coeffs);

        let p_3_coeffs = coeffs.iter().cloned().skip(3).step_by(2).collect_vec();
        let p_3_evals = dft.dft(p_3_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(p_0_evals, p_1_evals, p_2_evals, p_3_evals)
            .map(|(p_0_eval, p_1_eval, p_2_eval, p_3_eval)| p_0_eval + beta * p_1_eval + beta*beta * p_2_eval + beta*beta*beta * p_3_eval)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        reverse_slice_index_bits(&mut folded);
        folded = yu_fold_poly(folded, beta,4);
        reverse_slice_index_bits(&mut folded);

        assert_eq!(expected, folded);   
    }
}
