use std::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::TwoAdicField;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::{DenseMatrix, RowMajorMatrix};
use p3_matrix::{Dimensions, Matrix};
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::{info, info_span, instrument};

use p3_dft::TwoAdicSubgroupDft;

use crate::config::StirConfig;
use p3_fri::{prover::is_power_of_k, FriGenericConfig, LdtConfig, LdtProver};

use p3_fri::{naive_interpolation, vanishing_poly, Polynomial};

pub struct StirProver<'a, G, Val, Challenge, Challenger, M, FFT> {
    config: &'a StirConfig<M>,
    _marker: PhantomData<(G, Val, Challenge, Challenger, FFT)>,
}

impl<'a, G, Val, Challenge, M, Challenger, FFT> LdtProver<'a, G, Val, Challenge, M, Challenger>
    for StirProver<'a, G, Val, Challenge, Challenger, M, FFT>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
    FFT: TwoAdicSubgroupDft<Challenge>,
{
    type Conf = StirConfig<M>;
    type Proof = FinalProof<Challenge, M, Challenger::Witness>;
    fn new(g: &'a Self::Conf) -> Self {
        Self {
            config: g,
            _marker: PhantomData,
        }
    }

    fn folding_factor(&self) -> usize {
        1 << self.config.log_folding_factor()
    }

    fn prove(
        &self,
        g: &G,
        inputs: Vec<Vec<Challenge>>,
        challenger: &mut Challenger,
        open_input: impl Fn(usize) -> <G as FriGenericConfig<Challenge>>::InputProof,
    ) -> Self::Proof {
        // check sorted descending
        // assert!(inputs
        //     .iter()
        //     .tuple_windows()
        //     .all(|(l, r)| l.len() >= r.len()));

        let log_max_height = log2_strict_usize(inputs[0].len());

        // check the degree of input-polys is the power of folding_factor
        inputs.iter().for_each(|poly| {
            info!("poly_len: {:?}", poly.len());
            assert!(is_power_of_k(poly.len(), self.config.log_folding_factor()));
        });

        self.rounds(g, inputs, challenger, open_input)
    }
}

#[derive(Clone)]
pub struct FinalProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) commit: M::Commitment,
    pub(crate) rounds_proof: Vec<RoundProof<F, M, Witness>>,
    pub(crate) final_polynomial: Vec<F>,
    pub(crate) final_opening_proof: Vec<OpeningProof<F, M>>,
    pub(crate) pow_witness: Witness,
}

#[derive(Clone)]
pub struct OpeningProof<F: Field, M: Mmcs<F>> {
    // pub(crate) open_index: usize,
    pub(crate) open_row: Vec<F>,
    pub(crate) opening_proof: M::Proof,
}

#[derive(Clone)]
pub struct RoundProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) commit: M::Commitment, // prev-round poly commitment
    pub(crate) opening_proof: Vec<OpeningProof<F, M>>, // opening_proof for the prev-round poly
    pub(crate) betas: Vec<F>,
    pub(crate) ans_poly: Vec<F>,
    pub(crate) shake_poly: Vec<F>,
    pub(crate) pow_nonce: Witness,
}

impl<F: Field, M: Mmcs<F>, Witness> RoundProof<F, M, Witness> {
    pub fn check_merkle_path<'a>(&self, m: &'a M, dims: Dimensions, indexs: Vec<usize>) {
        self.opening_proof
            .iter()
            .zip(indexs.iter())
            .for_each(|(proof, index)| {
                m.verify_batch(
                    &self.commit,
                    &[dims],
                    *index,
                    &[proof.open_row.clone()],
                    &proof.opening_proof,
                )
                .expect("fail to verify open");
            });
    }
}

impl<'a, G, Val, Challenge, M, Challenger, FFT>
    StirProver<'a, G, Val, Challenge, Challenger, M, FFT>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
    FFT: TwoAdicSubgroupDft<Challenge>,
{
    fn rounds(
        &self,
        g: &G,
        inputs: Vec<Vec<Challenge>>,
        challenger: &mut Challenger,
        open_input: impl Fn(usize) -> <G as FriGenericConfig<Challenge>>::InputProof,
    ) -> FinalProof<Challenge, M, Challenger::Witness> {
        // we need to care about the w*w^2
        // we need to care about out of domian and inside of of domain sample

        let mut inputs_iter = inputs.into_iter().peekable();
        let mut folded: Vec<Challenge> = inputs_iter.next().unwrap();
        let mut commits: Vec<_> = vec![];
        let mut datas = vec![];
        let mut shifts = vec![];
        let mut generators = vec![];
        let mut folding_randomnesses = vec![];

        let folding_factor = 1 << self.config.log_folding_factor();
        let first_domain_bits = log2_ceil_usize(folded.len());
        let generator = Challenge::two_adic_generator(first_domain_bits);
        let mut new_generator = generator;

        let mut rounds_proof = vec![];
        for round in 0..self.config.num_rounds {
            shifts.push(generator.exp_u64(round as u64));
            new_generator = new_generator.exp_const_u64::<2>();
            generators.push(new_generator);
        }

        // first commit
        // w^0<w>     w^0 w^1 w^2 w^3 w^4 w^5 w^6 w^7
        let leaves = RowMajorMatrix::new(folded.clone(), folding_factor);
        let (commit, prover_data) = self.config.get_mmcs().commit_matrix(leaves);
        commits.push(commit.clone());
        datas.push(prover_data);
        challenger.observe(commit);
        let folding_randomness: Challenge = challenger.sample_ext_element();
        folding_randomnesses.push(folding_randomness);

        let dft_ins = FFT::default();
        let mut f_poly: Vec<Challenge> = dft_ins.idft(folded.clone());

        for round in 0..self.config.num_rounds - 1 {
            let (_cur_log_degree, _cur_log_blowup, cur_repetition) = self.config.rounds_info[round];

            // fold f polynomial
            let g_poly = Polynomial::from(f_poly.clone())
                .fold_coeff(folding_randomnesses[round].clone(), folding_factor);

            // commit matrix
            // w<w^2>     w^1 w^3 w^5 w^7   ==> scale-2 w^1 w^5 w<w^4>
            // w^2<w^4>   w^2 w^6
            // w^3<w^8>   w^3
            let g_evals = dft_ins.coset_dft(g_poly.values.clone(), shifts[round]);
            let leaves = RowMajorMatrix::new(g_evals, folding_factor);
            let (commit, prover_data) = self.config.get_mmcs().commit_matrix(leaves);
            challenger.observe(commit.clone());

            commits.push(commit);
            datas.push(prover_data);

            // out of domain sample
            let ood_randomness: Vec<Challenge> = (0..self.config.odd_samples)
                .map(|_| challenger.sample_ext_element())
                .collect();
            let betas: Vec<Challenge> = ood_randomness
                .iter()
                .map(|point| g_poly.evaluate(point.clone()))
                .collect();
            betas.iter().for_each(|beta| {
                challenger.observe_ext_element(beta.clone());
            });

            let comb_randomness: Challenge = challenger.sample_ext_element();
            let folding_randomness: Challenge = challenger.sample_ext_element();
            folding_randomnesses.push(folding_randomness);

            // Sample the indexes of L^k that we are going to use for querying the previous Merkle tree
            // Because we places the folding_factor evaluation at one leaf
            let prev_prover_data = &datas[round];
            let prev_commit = commits[round].clone();
            let leaves: &DenseMatrix<Challenge> = self
                .config
                .get_mmcs()
                .get_matrices(prev_prover_data)
                .pop()
                .unwrap();
            let log_height = log2_strict_usize(leaves.height());
            let stir_randomness_indexes: Vec<usize> = (0..cur_repetition)
                .into_iter()
                .map(|_| challenger.sample_bits(log_height))
                .collect();
            let open_proof: Vec<OpeningProof<Challenge, M>> = stir_randomness_indexes
                .iter()
                .map(|open_index| {
                    let (mut opened_rows, opening_proof) = self
                        .config
                        .get_mmcs()
                        .open_batch(*open_index, prev_prover_data);
                    assert_eq!(opened_rows.len(), 1);
                    let open_row = opened_rows.pop().unwrap();
                    assert_eq!(
                        open_row.len(),
                        folding_factor,
                        "the number of committed data should be euqal to folding_factor"
                    );
                    OpeningProof {
                        open_row,
                        opening_proof,
                    }
                })
                .collect();

            // pow
            let pow_witness = challenger.grind(self.config.param.pow_bits);

            let _shake_randomness: Challenge = challenger.sample_ext_element();

            let stir_randomness: Vec<Challenge> = stir_randomness_indexes
                .iter()
                .map(|index| {
                    // the generator need to exp the folding factor to eliminate the error that index is row-index instead of point-index
                    let scale_generator: Challenge =
                        shifts[round] * (generators[round].clone()).exp_u64(folding_factor as u64);
                    scale_generator.exp_u64(*index as u64)
                })
                .collect();

            // construct shake polynomial
            let betas_answer: Vec<(Challenge, Challenge)> = betas
                .iter()
                .zip(ood_randomness.iter())
                .map(|(beta, odd)| (*odd, *beta))
                .collect();

            let quotient_set: Vec<_> = ood_randomness
                .into_iter()
                .chain(stir_randomness.iter().cloned())
                .collect();

            let quotient_answers: Vec<(Challenge, Challenge)> = stir_randomness
                .iter()
                .map(|rand| (*rand, g_poly.evaluate(*rand)))
                .chain(betas_answer.clone().into_iter())
                .collect();

            // we can not use fft here, becuase the evaluation of betas, and the evaluations of stir_randomness are not in the same multiplicate subgroup
            let ans_polynomial = naive_interpolation(quotient_answers.iter());

            let mut shake_polynomial: Polynomial<Challenge> = Polynomial::from(vec![]);
            for (x, y) in quotient_answers {
                let num_polynomial = ans_polynomial.clone() - Polynomial::from(vec![y]);
                let den_polynomial = Polynomial::from(vec![-x, Challenge::one()]);
                shake_polynomial = shake_polynomial + (num_polynomial / den_polynomial);
            }

            // The quotient_polynomial is then computed
            let vanishing_poly = vanishing_poly(quotient_set.iter());
            // Resue the ans_polynomial to compute the quotient_polynomial
            let numerator = g_poly.clone() + ans_polynomial.clone();
            let quotient_polynomial = numerator / vanishing_poly;

            // This is the polynomial 1 + r * x + r^2 * x^2 + ... + r^n * x^n where n = |quotient_set|
            let scaling_polynomial = Polynomial::from(
                (0..quotient_set.len() + 1)
                    .map(|i| comb_randomness.exp_u64(i as u64))
                    .collect::<Vec<Challenge>>(),
            );

            let witness_polynomial = quotient_polynomial * scaling_polynomial;
            f_poly = witness_polynomial.values;

            let round_proof: RoundProof<Challenge, M, Challenger::Witness> = RoundProof {
                commit: prev_commit, // prev-round polynomial commitment
                opening_proof: open_proof,
                betas: betas,
                shake_poly: shake_polynomial.values,
                ans_poly: ans_polynomial.values,
                pow_nonce: pow_witness,
            };
            rounds_proof.push(round_proof);
        }

        assert_eq!(datas.len(), self.config.num_rounds);
        assert_eq!(commits.len(), self.config.num_rounds);
        assert_eq!(rounds_proof.len(), self.config.num_rounds - 1);

        let (_cur_log_degree, _cur_log_blowup, cur_repetition) =
            self.config.rounds_info[self.config.num_rounds - 1];

        // deal the final poly
        let final_polynomial = Polynomial::from(f_poly.clone()).fold_coeff(
            folding_randomnesses[self.config.num_rounds - 1],
            folding_factor,
        );

        let prev_prover_data = &datas[self.config.num_rounds - 1];
        let prev_commit = commits[self.config.num_rounds - 1].clone();
        let leaves: &DenseMatrix<Challenge> = self
            .config
            .get_mmcs()
            .get_matrices(prev_prover_data)
            .pop()
            .unwrap();
        let log_height = log2_strict_usize(leaves.height());
        let stir_randomness_indexes: Vec<usize> = (0..cur_repetition)
            .into_iter()
            .map(|_| challenger.sample_bits(log_height))
            .collect();
        let final_opening_proof: Vec<OpeningProof<Challenge, M>> = stir_randomness_indexes
            .iter()
            .map(|open_index| {
                let (mut opened_rows, opening_proof) = self
                    .config
                    .get_mmcs()
                    .open_batch(*open_index, prev_prover_data);
                assert_eq!(opened_rows.len(), 1);
                let open_row = opened_rows.pop().unwrap();
                assert_eq!(
                    open_row.len(),
                    folding_factor,
                    "the number of committed data should be euqal to folding_factor"
                );
                OpeningProof {
                    open_row,
                    opening_proof,
                }
            })
            .collect();

        // pow
        let pow_witness = challenger.grind(self.config.param.pow_bits);

        FinalProof {
            rounds_proof,
            final_polynomial: final_polynomial.values,
            commit: prev_commit,
            final_opening_proof,
            pow_witness,
        }
    }
}

mod tests {}
