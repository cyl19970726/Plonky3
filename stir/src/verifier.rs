use std::marker::PhantomData;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, Field, TwoAdicField};
use p3_fri::{
    verifier::FriError, CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, LdtConfig,
    LdtError, LdtVerifer, Polynomial, StirError,
};
use p3_matrix::Dimensions;

use crate::{config::StirConfig, prover::FinalProof};

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
            round_proof.check_merkle_path(self.config.get_mmcs(), dims, stir_randomness_indexes);

            if !challenger.check_witness(self.config.pow_bits(), proof.pow_witness) {
                return Err(StirError::InvalidPowWitness.into());
            }

            let shake_randomness: Challenge = challenger.sample_ext_element();

            let oracle_answer: Vec<Vec<Challenge>> = round_proof
                .opening_proof
                .iter()
                .map(|open_proof| open_proof.open_row.clone())
                .collect();
            let folded_answers = self.compute_folded_evaluations(folding_randomness, oracle_answer);

            let quotient_answers: Vec<(Challenge, Challenge)> = ood_randomness
                .into_iter()
                .zip(&round_proof.betas)
                .map(|(alpha, beta)| (alpha, *beta))
                .chain(folded_answers.into_iter())
                .collect();

            let interpolating_polynomial = round_proof.ans_poly.clone();
            let ans_eval = Polynomial::from(interpolating_polynomial).evaluate(shake_randomness);
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
        folding_randomness: Challenge,
        oracle_answers: Vec<Vec<Challenge>>,
    ) -> Vec<(Challenge, Challenge)> {
        unimplemented!()
    }
}
