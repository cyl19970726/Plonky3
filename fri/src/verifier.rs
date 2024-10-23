use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::Dimensions;

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, LdtVerifer};

pub struct Verifier<'a,G, Val, Challenge, M, Challenger>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    config: &'a FriConfig<M>,
    _marker: PhantomData<(G,Val,Challenger,Challenge)>,   
}

impl <'a,G, Val, Challenge, M, Challenger> LdtVerifer<'a,G,Val,Challenge,M,Challenger>for Verifier<'a,G, Val, Challenge, M, Challenger>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    type Proof = FriProof<Challenge, M, Challenger::Witness, G::InputProof>;
    fn folding_factor(&self) -> usize {
        self.config.folding_factor
    }

    fn new(config: &'a FriConfig<M>) -> Self {
        Self{
            config,
            _marker: PhantomData,
        }
    }

    fn verify(
            &self,
            g: &G,
            proof: &Self::Proof,
            challenger: &mut Challenger,
            open_input: impl Fn(usize, &G::InputProof) -> Result<Vec<(usize, Challenge)>, G::InputError>,
        ) -> Result<(),FriError<M::Error, G::InputError>> {
        verify(g, self.config, proof, challenger, open_input)
    }
}
#[derive(Debug)]
pub enum FriError<CommitMmcsErr, InputError> {
    InvalidProofShape,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    InvalidPowWitness,
}

pub fn verify<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    proof: &FriProof<Challenge, M, Challenger::Witness, G::InputProof>,
    challenger: &mut Challenger,
    open_input: impl Fn(usize, &G::InputProof) -> Result<Vec<(usize, Challenge)>, G::InputError>,
) -> Result<(), FriError<M::Error, G::InputError>>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    let betas: Vec<Challenge> = proof
        .commit_phase_commits
        .iter()
        .map(|comm| {
            challenger.observe(comm.clone());
            challenger.sample_ext_element()
        })
        .collect();
    challenger.observe_ext_element(proof.final_poly);

    if proof.query_proofs.len() != config.num_queries {
        return Err(FriError::InvalidProofShape);
    }

    // Check PoW.
    if !challenger.check_witness(config.proof_of_work_bits, proof.pow_witness) {
        return Err(FriError::InvalidPowWitness);
    }

    let log_max_height = proof.commit_phase_commits.len() * config.log_folding_factor + config.log_blowup;

    tracing::info!("verifier log_max_height {:?}",log_max_height);

    for qp in &proof.query_proofs {
        let index = challenger.sample_bits(log_max_height + g.extra_query_index_bits());
        let ro = open_input(index, &qp.input_proof).map_err(FriError::InputError)?;

        debug_assert!(
            ro.iter().tuple_windows().all(|((l, _), (r, _))| l > r),
            "reduced openings sorted by height descending"
        );

        let folded_eval = verify_query(
            g,
            config,
            index >> g.extra_query_index_bits(),
            izip!(
                &betas,
                &proof.commit_phase_commits,
                &qp.commit_phase_openings
            ),
            ro,
            log_max_height,
        )?;

        if folded_eval != proof.final_poly {
            return Err(FriError::FinalPolyMismatch);
        }
    }

    Ok(())
}

type CommitStep<'a, F, M> = (
    &'a F,
    &'a <M as Mmcs<F>>::Commitment,
    &'a CommitPhaseProofStep<F, M>,
);

fn verify_query<'a, G, F, M>(
    g: &G,
    config: &FriConfig<M>,
    mut index: usize,
    steps: impl Iterator<Item = CommitStep<'a, F, M>>,
    reduced_openings: Vec<(usize, F)>,
    log_max_height: usize,
) -> Result<F, FriError<M::Error, G::InputError>>
where
    F: Field,
    M: Mmcs<F> + 'a,
    G: FriGenericConfig<F>,
{
    let mut folded_eval = F::zero();
    let mut ro_iter = reduced_openings.into_iter().peekable();
    for i in ro_iter.clone() {
        tracing::info!("ro_iter {:?}",i);
    }
    let mut times = 0;
    for (log_folded_height, (&beta, comm, opening)) in izip!((0..(log_max_height - (config.log_folding_factor - 1))).rev().step_by(config.log_folding_factor), steps) {
        tracing::info!("verifier log_max_height {:?}", log_max_height);
        tracing::info!("prev_folded_eval {:?}",folded_eval);
        tracing::info!("log_folded_height {:?}",log_folded_height);
        if let Some((_, ro)) = ro_iter.next_if(|(lh, _)| *lh == log_folded_height + config.log_folding_factor) {
            tracing::info!("ro:{:?}", ro);
            folded_eval += ro;
        }

        tracing::info!("verifier query times {:?}",times);
        times += 1;

        // let index_sibling = index ^ 1;
        let index_pair = index >> config.log_folding_factor;

        // check the folded_eval from the leaf        
        // let mut evals = vec![folded_eval; 2];
        let mut valid_folded_value = false;
        // let mut open_point_index = 0;
        let opening_row = opening.opened_row.clone();

        tracing::info!("opening_row{:?}",opening_row);
        tracing::info!("folded_eval {:?}",folded_eval);

        for i in 0..opening_row.len(){
            if opening_row[i] == folded_eval{
                valid_folded_value = true;
                // open_point_index = i;
                break;
            }
        }
        
        assert!(valid_folded_value);

        // evals[index_sibling % 2] = opening.sibling_value;

        let dims = &[Dimensions {
            width: config.folding_factor,
            height: 1 << log_folded_height,
        }];
        config
            .mmcs
            .verify_batch(
                comm,
                dims,
                index_pair,
                &[opening_row.clone()],
                &opening.opening_proof,
            )
            .map_err(FriError::CommitPhaseMmcsError)?;

        index = index_pair;

        folded_eval = g.fold_row(index, log_folded_height, beta, opening_row.into_iter(),config.folding_factor);
    }

    debug_assert!(index < config.blowup(), "index was {}", index);
    debug_assert!(
        ro_iter.next().is_none(),
        "verifier reduced_openings were not in descending order?"
    );

    Ok(folded_eval)
}
