use std::marker::PhantomData;

use crate::{config::StirConfig, prover::FinalProof};
use itertools::izip;
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_fri::{
    lagrange_interpolate_and_evaluate, verifier::FriError, CommitPhaseProofStep, FriConfig,
    FriGenericConfig, FriProof, LdtConfig, LdtError, LdtVerifer, Polynomial, StirError,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::Dimensions;
use p3_util::reverse_slice_index_bits;
pub struct StirVerifier<'a, G, Val, Challenge, Challenger, M, FFT>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    config: &'a StirConfig<M>,
    _marker: PhantomData<(G, Val, Challenge, Challenger, FFT)>,
}

impl<'a, G, Val, Challenge, M, Challenger, FFT> LdtVerifer<'a, G, Val, Challenge, M, Challenger>
    for StirVerifier<'a, G, Val, Challenge, Challenger, M, FFT>
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

    fn new(config: &'a Self::Conf) -> Self {
        Self {
            config: config,
            _marker: PhantomData,
        }
    }

    fn folding_factor(&self) -> usize {
        1 << self.config.log_folding_factor()
    }

    fn verify(
        &self,
        g: &G,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        open_input: impl Fn(
            usize,
            &<G as FriGenericConfig<Challenge>>::InputProof,
        ) -> Result<
            Vec<(usize, Challenge)>,
            <G as FriGenericConfig<Challenge>>::InputError,
        >,
    ) -> Result<
        (),
        p3_fri::LdtError<
            <M as Mmcs<Challenge>>::Error,
            <G as FriGenericConfig<Challenge>>::InputError,
        >,
    > {
        if proof.final_polynomial.len() != 1 {
            return Err(StirError::FinalPolyMismatch.into());
        }

        let first_commit = proof.rounds_proof[0].commit.clone();
        challenger.observe(first_commit);
        let folding_randomness: Challenge = challenger.sample_ext_element();

        let domain_gen = Challenge::two_adic_generator(self.config.log_start_degree());
        let domain_size = 1 << self.config.log_start_degree();

        let mut verification_state = VerificationState {
            oracle: OracleType::Initial,
            domain_gen,
            domain_size,
            domain_offset: Challenge::one(),
            root_of_unity: domain_gen,
            num_round: 0,
            folding_randomness,
        };

        for round in 0..self.config.num_rounds - 1 {
            let round_proof = proof.rounds_proof[round].clone();
            let (cur_log_degree, _cur_log_blowup, cur_repetition) = self.config.rounds_info[round];
            challenger.observe(proof.rounds_proof[round + 1].commit.clone());

            let ood_randomness: Vec<Challenge> = (0..self.config.odd_samples)
                .map(|_| challenger.sample_ext_element())
                .collect();
            round_proof.betas.iter().for_each(|beta| {
                challenger.observe_ext_element(beta.clone());
            });

            let comb_randomness: Challenge = challenger.sample_ext_element();
            let folding_randomness: Challenge = challenger.sample_ext_element();

            let log_height = cur_log_degree >> self.config.log_folding_factor();
            let stir_randomness_indexes: Vec<usize> = (0..cur_repetition)
                .into_iter()
                .map(|_| challenger.sample_bits(log_height))
                .collect();
            let dims = Dimensions {
                width: 1 << self.config.log_folding_factor(),
                height: 1 << log_height,
            };
            round_proof.check_merkle_path(
                self.config.get_mmcs(),
                dims,
                stir_randomness_indexes.clone(),
            );

            if !challenger.check_witness(self.config.pow_bits(), proof.pow_witness) {
                return Err(StirError::InvalidPowWitness.into());
            }

            let shake_randomness: Challenge = challenger.sample_ext_element();

            let oracle_answer: Vec<Vec<Challenge>> = round_proof
                .opening_proof
                .iter()
                .map(|open_proof| {
                    let mut row = open_proof.open_row.clone();
                    reverse_slice_index_bits(&mut row);
                    row
                })
                .collect();

            let folded_answers = self.compute_folded_evaluations(
                stir_randomness_indexes,
                oracle_answer,
                &verification_state,
            );

            let quotient_answers: Vec<(Challenge, Challenge)> = ood_randomness
                .into_iter()
                .zip(&round_proof.betas)
                .map(|(alpha, beta)| (alpha, *beta))
                .chain(folded_answers.into_iter())
                .collect();

            let interpolating_polynomial = round_proof.ans_poly.clone();
            let ans_eval =
                Polynomial::from(interpolating_polynomial.clone()).evaluate(shake_randomness);
            let shake_eval =
                Polynomial::from(round_proof.shake_poly.clone()).evaluate(shake_randomness);

            let mut denoms: Vec<Challenge> = quotient_answers
                .iter()
                .map(|(x, _)| shake_randomness - *x)
                .collect();

            if shake_eval
                != quotient_answers
                    .iter()
                    .zip(denoms)
                    .map(|((_, y), d)| (ans_eval - *y) * d)
                    .sum()
            {
                return Err(p3_fri::StirError::InvalidShake.into());
            }

            let quotient_set = quotient_answers
                .into_iter()
                .map(|(x, _)| x)
                .collect::<Vec<_>>();

            verification_state = VerificationState {
                oracle: OracleType::Virtual(VirtualFunction {
                    comb_randomness,
                    quotient_set,
                    interpolating_polynomial,
                }),
                // TODO: We can optimize
                domain_size: verification_state.domain_size / 2,
                domain_gen: verification_state.domain_gen * verification_state.domain_gen,
                domain_offset: verification_state.domain_offset
                    * verification_state.domain_offset
                    * verification_state.root_of_unity,
                root_of_unity: verification_state.root_of_unity,
                folding_randomness: folding_randomness,
                num_round: verification_state.num_round + 1,
            }
        }

        // final round
        let (final_log_degree, final_log_blowup, final_repetition) =
            self.config.rounds_info[self.config.num_rounds];

        let log_height = final_log_degree >> self.config.log_folding_factor();
        let stir_randomness_indexes: Vec<usize> = (0..final_repetition)
            .into_iter()
            .map(|_| challenger.sample_bits(log_height))
            .collect();

        if !challenger.check_witness(self.config.pow_bits(), proof.pow_witness) {
            return Err(StirError::InvalidPowWitness.into());
        }

        let oracle_answer: Vec<Vec<Challenge>> = proof
            .final_opening_proof
            .iter()
            .map(|open_proof| {
                let mut row = open_proof.open_row.clone();
                reverse_slice_index_bits(&mut row);
                row
            })
            .collect();

        let folded_answers = self.compute_folded_evaluations(
            stir_randomness_indexes,
            oracle_answer,
            &verification_state,
        );

        let sucess = folded_answers.into_iter().all(|(point, value)| {
            Polynomial::from(proof.final_polynomial.clone()).evaluate(point) == value
        });
        if !sucess {
            return Err(StirError::FinalPolyCheckErr.into());
        }
        Ok(())
    }
}

impl<'a, G, Val, Challenge, M, Challenger, FFT>
    StirVerifier<'a, G, Val, Challenge, Challenger, M, FFT>
where
    Val: Field,
    Challenge: ExtensionField<Val> + TwoAdicField,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
    FFT: TwoAdicSubgroupDft<Challenge>,
{
    fn compute_folded_evaluations(
        &self,
        stir_randomness_indexes: Vec<usize>,
        oracle_answers: Vec<Vec<Challenge>>,
        verification_state: &VerificationState<Challenge>,
    ) -> Vec<(Challenge, Challenge)> {
        let folding_factor = 1 << self.config.log_folding_factor();
        let scaling_factor =
            verification_state.domain_size / (1 << self.config.log_folding_factor());

        let generator = verification_state.domain_gen.exp_u64(scaling_factor as u64);

        let coset_offsets: Vec<Challenge> = stir_randomness_indexes
            .iter()
            .map(|stir_randomness_index| {
                generator.exp_u64(*stir_randomness_index as u64)
                    * verification_state.domain_offset.clone()
            })
            .collect();

        let scales: Vec<Challenge> =
            std::iter::successors(Some(Challenge::one()), |&prev| Some(prev * generator))
                .take(folding_factor)
                .collect();

        let query_sets: Vec<Vec<Challenge>> = coset_offsets
            .iter()
            .map(|coset_offset| {
                (0..folding_factor)
                    .map(|j| *coset_offset * scales[j])
                    .collect::<Vec<_>>()
            })
            .collect();

        let common_factor_scale = match &verification_state.oracle {
            OracleType::Initial => Challenge::zero(),
            OracleType::Virtual(virtual_function) => virtual_function.comb_randomness,
        };

        let global_common_factors: Vec<Vec<Challenge>> = query_sets
            .iter()
            .map(|query_set| {
                query_set
                    .iter()
                    .map(|x| Challenge::one() - common_factor_scale * x.clone())
                    .collect::<Vec<Challenge>>()
            })
            .collect();

        let common_factors_inv: Vec<Vec<Challenge>> = global_common_factors
            .iter()
            .map(|common_factor_set| {
                common_factor_set
                    .iter()
                    .map(|common_factor: &Challenge| (*common_factor).inverse())
                    .collect::<Vec<Challenge>>()
            })
            .collect();

        let global_denominators: Vec<Vec<Challenge>> = query_sets
            .iter()
            .map(|query_set| match &verification_state.oracle {
                OracleType::Initial => vec![Challenge::one(); query_set.len()],
                OracleType::Virtual(virtual_function) => query_set
                    .iter()
                    .map(|eval_point| {
                        virtual_function
                            .quotient_set
                            .iter()
                            .map(|x| *eval_point - *x)
                            .product::<Challenge>()
                    })
                    .collect::<Vec<_>>(),
            })
            .collect();
        let denominators_inv: Vec<Vec<Challenge>> = global_denominators
            .iter()
            .map(|global_denominator_set| {
                global_denominator_set
                    .iter()
                    .map(|denominator| (*denominator).inverse())
                    .collect()
            })
            .collect();

        let coset_offsets_inv: Vec<Challenge> = coset_offsets
            .iter()
            .map(|offset| offset.inverse())
            .collect();

        let dft = Radix2Dit::default();

        let evaluations_of_ans: Vec<Vec<Challenge>> = coset_offsets
            .iter()
            .zip(&coset_offsets_inv)
            .map(
                |(coset_offset, coset_offset_inv)| match &verification_state.oracle {
                    OracleType::Initial => vec![Challenge::one(); folding_factor],
                    OracleType::Virtual(virtual_function) => {
                        // todo: bad implementation
                        dft.coset_dft(
                            virtual_function.interpolating_polynomial.clone(),
                            coset_offset.clone(),
                        )
                    }
                },
            )
            .collect();

        let scaled_offset = verification_state
            .domain_offset
            .exp_u64(folding_factor as u64);

        izip!(
            stir_randomness_indexes.iter(),
            coset_offsets,
            coset_offsets_inv,
            query_sets,
            common_factors_inv,
            denominators_inv,
            evaluations_of_ans
        )
        .enumerate()
        .map(
            |(
                i,
                (
                    stir_randomness_index,
                    coset_offset,
                    coset_offset_inv,
                    query_set,
                    common_factors_inv,
                    denominators_inv,
                    evaluation_of_ans,
                ),
            )| {
                // This is the point that we are querying at
                let stir_randomness = scaled_offset
                    * verification_state
                        .domain_gen
                        .exp_u64((folding_factor * stir_randomness_index) as u64);

                let f_answers: Vec<_> = query_set
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(j, x)| {
                        verification_state.query(
                            x,
                            oracle_answers[i][j],
                            common_factors_inv[j],
                            denominators_inv[j],
                            evaluation_of_ans[j],
                        )
                    })
                    .collect();

                // This is the folding
                let folded_answer = lagrange_interpolate_and_evaluate(
                    &query_set,
                    &f_answers,
                    verification_state.folding_randomness,
                );

                // Return the folded answer
                (stir_randomness, folded_answer)
            },
        )
        .collect()
    }
}

#[derive(Debug)]
pub struct VirtualFunction<F: TwoAdicField + Field> {
    comb_randomness: F,
    interpolating_polynomial: Vec<F>,
    quotient_set: Vec<F>,
}

#[derive(Debug)]
pub enum OracleType<F: TwoAdicField + Field> {
    Initial,
    Virtual(VirtualFunction<F>),
}

#[derive(Debug)]
pub struct VerificationState<F: TwoAdicField + Field> {
    oracle: OracleType<F>,
    domain_gen: F,
    domain_size: usize,
    domain_offset: F,
    root_of_unity: F,
    folding_randomness: F,
    num_round: usize,
}

impl<F: TwoAdicField + Field> VerificationState<F> {
    pub fn query(
        &self,
        evaluation_point: F,
        value_of_prev_oracle: F,
        common_factors_inverse: F,
        denom_hint: F,
        ans_eval: F,
    ) -> F {
        match &self.oracle {
            OracleType::Initial => value_of_prev_oracle,
            OracleType::Virtual(virtual_function) => {
                let num_terms = virtual_function.quotient_set.len();

                // quotient with hint
                let quotient_evaluation = quotient_with_hint(
                    value_of_prev_oracle,
                    evaluation_point,
                    &virtual_function.quotient_set,
                    denom_hint,
                    ans_eval,
                );

                let common_factor = evaluation_point * virtual_function.comb_randomness;

                let scale_factor = if common_factor != F::one() {
                    (F::one() - common_factor.exp_u64((num_terms + 1) as u64))
                        * common_factors_inverse
                } else {
                    F::from_canonical_u64((num_terms + 1) as u64)
                };

                quotient_evaluation * scale_factor
            }
        }
    }
}

// Allows to amortize the evaluation of the quotient polynomial
pub fn quotient_with_hint<'a, F: Field>(
    claimed_eval: F,
    evaluation_point: F,
    quotient_set: impl IntoIterator<Item = &'a F>,
    //ans_polynomial: &DensePolynomial<F>,
    denom_hint: F,
    ans_eval: F,
) -> F {
    let quotient_set: Vec<_> = quotient_set.into_iter().copied().collect();

    // Check if the evaluation point is in the domain
    for dom in quotient_set.iter() {
        if evaluation_point == *dom {
            panic!("Evaluation point is in the domain");
        }
    }

    let num = claimed_eval - ans_eval;

    num * denom_hint
}
