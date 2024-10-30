use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::TwoAdicField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_bits_len, reverse_slice_index_bits};
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

pub fn fold_poly<F: TwoAdicField,M: Matrix<F>>(poly: Vec<F>, beta: F, folding_factor: usize) -> Vec<F> {
    assert!(folding_factor > 0 && (folding_factor & (folding_factor - 1)) == 0, "The folding factor must be a power of 2");
    assert!(poly.len() % folding_factor == 0, "The length of the poly must be divisible by the folding factor");

    // let mut xs = F::two_adic_generator(log2_strict_usize(poly.len())).powers().take(poly.len()).collect::<Vec<F>>();

    // reverse_slice_index_bits(&mut xs);
    // let xs_matrix = RowMajorMatrix::new(xs,folding_factor);
    let m = RowMajorMatrix::new(poly, folding_factor);
    fold_poly_matrix(m, beta, folding_factor)
    // Parallel processing and caching beta powers
    // m.row_slices().zip(xs_matrix.row_slices()).map(|(eval_row,xs_row)| {
    //     lagrange_interpolate_and_evaluate(xs_row,eval_row,beta)
    // }).collect::<Vec<F>>()
}

pub fn fold_poly_matrix<F: TwoAdicField,M: Matrix<F>>(poly: M, beta: F, folding_factor: usize) -> Vec<F> {

    let poly_degree = poly.width() * poly.height();
    let mut xs = F::two_adic_generator(log2_strict_usize(poly_degree)).powers().take(poly_degree).collect::<Vec<F>>();

    reverse_slice_index_bits(&mut xs);
    let xs_matrix = RowMajorMatrix::new(xs,folding_factor);

    // Parallel processing and caching beta powers
    poly.to_row_major_matrix().row_slices().zip(xs_matrix.row_slices()).map(|(eval_row,xs_row)| {
        
        lagrange_interpolate_and_evaluate(xs_row,eval_row,beta)
    }).collect::<Vec<F>>()
}

pub fn yu_fold_poly<F: TwoAdicField>(poly: Vec<F>, beta: F, folding_factor: usize) -> Vec<F> {
    assert!(folding_factor > 0 && (folding_factor & (folding_factor - 1)) == 0, "The folding factor must be a power of 2");
    assert!(poly.len() % folding_factor == 0, "The length of the poly must be divisible by the folding factor");
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

pub fn fold_poly_with_dft<F: TwoAdicField>(poly: Vec<F>, beta: F, folding_factor: usize) -> Vec<F> {
    assert!(folding_factor > 0 && (folding_factor & (folding_factor - 1)) == 0, "The folding factor must be a power of 2");
    assert!(poly.len() % folding_factor == 0, "The length of the poly must be divisible by the folding factor");

    let m = RowMajorMatrix::new(poly, folding_factor);

    let g_inv = F::two_adic_generator(log2_strict_usize(m.height()) + log2_strict_usize(folding_factor)).inverse();

    //beta * g_inv^i
    let mut xs = g_inv
        .shifted_powers(beta)
        .take(m.height())
        .collect_vec();

    reverse_slice_index_bits(&mut xs);

    let dft = Radix2Dit::default();

    // Parallel processing and caching beta powers
    let mut row_count = 0;
    m.row_slices().map(|row| {
        let mut row_evals = row.to_vec();
        reverse_slice_index_bits(&mut row_evals);

        let row_coeff = dft.idft(row_evals.clone());

        let x = xs[row_count];
        row_count += 1;

        row_coeff.iter().enumerate().fold(F::zero(), |acc, (power, coeff)| {
            acc + (*coeff * x.exp_u64(power as u64))
        })
    }).collect::<Vec<F>>()
}


pub fn fold_row<F: TwoAdicField>(
    index: usize,
    log_height: usize,
    beta: F,
    evals: impl Iterator<Item = F>,
    folding_factor: usize,
) -> F {
    let arity = folding_factor;
    let log_arity = log2_strict_usize(folding_factor);
    let (e0, e1) = evals
        .collect_tuple()
        .expect("TwoAdicFriFolder only supports arity=2");
    // If performance critical, make this API stateful to avoid this
    // This is a bit more math than is necessary, but leaving it here
    // in case we want higher arity in the future
    let subgroup_start = F::two_adic_generator(log_height + log_arity)
        .exp_u64(reverse_bits_len(index, log_height) as u64);
    let mut xs = F::two_adic_generator(log_arity)
        .shifted_powers(subgroup_start)
        .take(arity)
        .collect_vec();
    reverse_slice_index_bits(&mut xs);
    e0 + (beta - xs[0]) * (e1 - e0) / (xs[1] - xs[0])
}

// Assuming a field F that supports arithmetic operations.
pub fn multi_fold_row<F: TwoAdicField>(
    index: usize,
    log_height: usize,
    beta: F,
    evals: impl Iterator<Item = F>,
    folding_factor: usize,  // folding_factor now supports 2, 4, 8, 16, etc.
) -> F {
    let arity = folding_factor;
    let log_arity = log2_strict_usize(folding_factor);

    // Collect all the evaluation points (arity number of points)
    let eval_points: Vec<F> = evals.take(arity).collect();
    assert_eq!(eval_points.len(), arity, "Expected evals to match the arity");

    // Compute the subgroup start as before
    let subgroup_start = F::two_adic_generator(log_height + log_arity)
        .exp_u64(reverse_bits_len(index, log_height) as u64);
    
    // Compute xs values (powers of the generator for interpolation points)
    let mut xs = F::two_adic_generator(log_arity)
        .shifted_powers(subgroup_start)
        .take(arity)
        .collect_vec();
    reverse_slice_index_bits(&mut xs);

    // Perform Lagrange interpolation over the arity points
    lagrange_interpolate_and_evaluate(&xs, &eval_points, beta)
}

// Helper function to perform Lagrange interpolation
fn lagrange_interpolate_and_evaluate<F: TwoAdicField>(
    xs: &[F],            // The x values (like powers of the two-adic generator)
    evals: &[F],          // The evaluation points corresponding to xs
    beta: F,              // The point where we want to evaluate the interpolated polynomial
) -> F {
    let mut result = F::zero();
    let n = xs.len();

    // Lagrange interpolation formula:
    // P(beta) = Σ (y_i * l_i(beta))
    // Where l_i(beta) = Π (beta - x_j) / (x_i - x_j) for all j != i
    for i in 0..n {
        let mut li = F::one();
        for j in 0..n {
            if i != j {
                li = li * (beta - xs[j]) / (xs[i] - xs[j]);
            }
        }
        result = result + evals[i] * li;
    }
    
    result
}


#[cfg(test)]
mod tests {
    use alloc::{collections::btree_map::Range, vec};
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
        let first_row =&folded[4..6].to_vec();
        let folded_row_res = fold_row(2, log_n, beta, first_row.iter().cloned(), 2);

        let folded_row_res1 = multi_fold_row(2, log_n, beta, first_row.iter().cloned(), 2);
        folded = fold_even_odd(folded, beta);

        assert_eq!(folded[2],folded_row_res);
        assert_eq!(folded_row_res1,folded_row_res);
        reverse_slice_index_bits(&mut folded);
        assert_eq!(expected, folded);
        // fold_row(index, log_height, beta, evals, folding_factor)
    }

    #[test]
    fn test_fold_poly(){
        type F = BabyBear;

        let mut rng = thread_rng();
        let folding_factor = 4;

        let log_n = 4;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let p_0_coeffs = coeffs.iter().cloned().step_by(folding_factor).collect_vec();
        let new_degree  = p_0_coeffs.len();
        let p_0_evals = dft.dft(p_0_coeffs);
        
        let p_1_coeffs = coeffs.iter().cloned().skip(1).step_by(folding_factor).collect_vec();
        let p_1_evals = dft.dft(p_1_coeffs);

        let p_2_coeffs = coeffs.iter().cloned().skip(2).step_by(folding_factor).collect_vec();
        let p_2_evals = dft.dft(p_2_coeffs);

        let p_3_coeffs = coeffs.iter().cloned().skip(3).step_by(folding_factor).collect_vec();
        let p_3_evals = dft.dft(p_3_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(p_0_evals, p_1_evals, p_2_evals, p_3_evals)
            .map(|(p_0_eval, p_1_eval, p_2_eval, p_3_eval)| p_0_eval + beta * p_1_eval + beta*beta * p_2_eval + beta*beta*beta * p_3_eval)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        let mut new_folded = vec![];
        reverse_slice_index_bits(&mut folded);
        for i in 0..10{
            let sample_index = rng.gen::<usize>() % new_degree;
            let range = sample_index * folding_factor..(sample_index+1) * folding_factor;
            let sample_row =&folded[range].to_vec();
            let multi_fold_row_res = multi_fold_row(sample_index, log_n, beta, sample_row.into_iter().cloned(), folding_factor);
            new_folded = fold_poly::<F,RowMajorMatrix<F>>(folded.clone(), beta,folding_factor);
            assert_eq!(new_folded[sample_index],multi_fold_row_res);
        }

        reverse_slice_index_bits(&mut new_folded);

        assert_eq!(expected, new_folded);   
    }

    #[test]
    fn test_fold_poly_dft(){
        type F = BabyBear;

        let mut rng = thread_rng();
        let folding_factor = 4;

        let log_n = 4;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let p_0_coeffs = coeffs.iter().cloned().step_by(folding_factor).collect_vec();
        let new_degree  = p_0_coeffs.len();
        let p_0_evals = dft.dft(p_0_coeffs);

        let p_1_coeffs = coeffs.iter().cloned().skip(1).step_by(folding_factor).collect_vec();
        let p_1_evals = dft.dft(p_1_coeffs);

        let p_2_coeffs = coeffs.iter().cloned().skip(2).step_by(folding_factor).collect_vec();
        let p_2_evals = dft.dft(p_2_coeffs);

        let p_3_coeffs = coeffs.iter().cloned().skip(3).step_by(folding_factor).collect_vec();
        let p_3_evals = dft.dft(p_3_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(p_0_evals, p_1_evals, p_2_evals, p_3_evals)
            .map(|(p_0_eval, p_1_eval, p_2_eval, p_3_eval)| p_0_eval + beta * p_1_eval + beta*beta * p_2_eval + beta*beta*beta * p_3_eval)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        let mut new_folded = vec![];
        reverse_slice_index_bits(&mut folded);
        for i in 0..10{
            let sample_index = rng.gen::<usize>() % new_degree;
            let range = sample_index * folding_factor..(sample_index+1) * folding_factor;
            let sample_row =&folded[range].to_vec();
            let multi_fold_row_res = multi_fold_row(sample_index, log_n, beta, sample_row.into_iter().cloned(), folding_factor);
            new_folded = fold_poly::<F,RowMajorMatrix<F>>(folded.clone(), beta,folding_factor);
            assert_eq!(new_folded[sample_index],multi_fold_row_res);
        }

        reverse_slice_index_bits(&mut new_folded);

        assert_eq!(expected, new_folded);   
    }

    #[test]
    fn test_yu_fold_poly(){
        type F = BabyBear;
        let folding_factor = 4;

        let mut rng = thread_rng();

        let log_n = 4;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let p_0_coeffs = coeffs.iter().cloned().step_by(folding_factor).collect_vec();
        let new_degree  = p_0_coeffs.len();
        let p_0_evals = dft.dft(p_0_coeffs);

        let p_1_coeffs = coeffs.iter().cloned().skip(1).step_by(folding_factor).collect_vec();
        let p_1_evals = dft.dft(p_1_coeffs);

        let p_2_coeffs = coeffs.iter().cloned().skip(2).step_by(folding_factor).collect_vec();
        let p_2_evals = dft.dft(p_2_coeffs);

        let p_3_coeffs = coeffs.iter().cloned().skip(3).step_by(folding_factor).collect_vec();
        let p_3_evals = dft.dft(p_3_coeffs);

        let beta = rng.gen::<F>();
        let expected = izip!(p_0_evals, p_1_evals, p_2_evals, p_3_evals)
            .map(|(p_0_eval, p_1_eval, p_2_eval, p_3_eval)| p_0_eval + beta * p_1_eval + beta*beta * p_2_eval + beta*beta*beta * p_3_eval)
            .collect::<Vec<_>>();

        // fold_even_odd takes and returns in bitrev order.
        let mut folded = evals;
        let mut new_folded = vec![];
        reverse_slice_index_bits(&mut folded);
        for i in 0..10{
            let sample_index = rng.gen::<usize>() % new_degree;
            let range = sample_index * folding_factor..(sample_index+1) * folding_factor;
            let sample_row =&folded[range].to_vec();
            let multi_fold_row_res = multi_fold_row(sample_index, log_n, beta, sample_row.into_iter().cloned(), folding_factor);
            new_folded = fold_poly::<F,RowMajorMatrix<F>>(folded.clone(), beta,folding_factor);
            assert_eq!(new_folded[sample_index],multi_fold_row_res);
        }

        reverse_slice_index_bits(&mut new_folded);

        assert_eq!(expected, new_folded);   
    }
}
