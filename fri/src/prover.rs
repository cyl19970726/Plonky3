use alloc::vec;
use alloc::vec::Vec;
use core::{iter, marker::PhantomData};

use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info, info_span, instrument};

use crate::{CommitPhaseProofStep, FriConfig, FriGenericConfig, FriProof, LdtProver, QueryProof};

pub struct FriProver<'a,G, Val, Challenge, M, Challenger> 
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>
{
    config: &'a FriConfig<M>,
    _marker: PhantomData<(G,Val,M,Challenger,Challenge)>
}

impl <'a,G, Val, Challenge, M, Challenger>  LdtProver<'a,G,Val,Challenge,M,Challenger>  for FriProver<'a,G,Val,Challenge,M,Challenger> 
where 
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>
{
    type Proof = FriProof<Challenge,M, Challenger::Witness, G::InputProof>;

    fn new(config: &'a FriConfig<M>) -> Self{
        Self{
            config,
            _marker: PhantomData,
        }
    }

    fn folding_factor(&self) -> usize {
        self.config.folding_factor
    }

    fn prove(&self,   
            g: &G,
            inputs: Vec<Vec<Challenge>>,
            challenger: &mut Challenger,
            open_input: impl Fn(usize) -> G::InputProof
        ) -> Self::Proof {
        prove(g, self.config, inputs, challenger, open_input)    
    }
}

/// This function checks if a given `value` is a power of `k`.
/// It returns `true` if `value` is a power of `k`, otherwise `false`.
pub fn is_power_of_k(degree_bits: usize, log_k: usize) -> bool {
    degree_bits % log_k == 0
}
#[cfg(test)]
mod test{
    use super::is_power_of_k;

    #[test]
    fn test_pk(){
        let folding_factor = 4;//log = 2
        let value=  4<<1;
        let valid = is_power_of_k(value, folding_factor);
        assert!(valid)
    }
}

// polynomial commitment 
// PCS 接口 
// FRI --> query_nums 0..16 
//  for  0...16 { n degree fold into const } 
// Stir  F --> Fold  --> g_poly --> query_num --> 计算挑战点 r_out, r_shift --> g_poly eval --> ans(x) --> shake_poly --> quotient(x) --> degree_correct --> next f 
#[instrument(name = "FRI prover", skip_all)]
pub fn prove<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>, // 2 4 8 16     // 16 4 
    challenger: &mut Challenger,
    open_input: impl Fn(usize) -> G::InputProof,
) -> FriProof<Challenge, M, Challenger::Witness, G::InputProof>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    // check sorted descending
    assert!(inputs
        .iter()
        .tuple_windows()
        .all(|(l, r)| l.len() >= r.len()));

    let log_max_height = log2_strict_usize(inputs[0].len());

    // make sure that the degree of inputs is the power of folding_factor 
    inputs.iter().for_each(|poly|{
        info!("poly_len: {:?}", poly.len());

        assert!(is_power_of_k(poly.len(), config.folding_factor));
        
    });

    let commit_phase_result = commit_phase(g, config, inputs, challenger);

    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_proofs = info_span!("query phase").in_scope(|| {
        iter::repeat_with(|| challenger.sample_bits(log_max_height + g.extra_query_index_bits()))
            .take(config.num_queries)
            .map(|index| QueryProof {
                input_proof: open_input(index),
                commit_phase_openings: answer_query(
                    config,
                    &commit_phase_result.data,
                    index >> g.extra_query_index_bits(),
                ),
            })
            .collect()
    });

    FriProof {
        commit_phase_commits: commit_phase_result.commits,
        query_proofs,
        final_poly: commit_phase_result.final_poly,
        pow_witness,
    }
}

struct CommitPhaseResult<F: Field, M: Mmcs<F>> {
    commits: Vec<M::Commitment>,
    data: Vec<M::ProverData<RowMajorMatrix<F>>>,
    final_poly: F,
}

#[instrument(name = "commit phase", skip_all)]
fn commit_phase<G, Val, Challenge, M, Challenger>(
    g: &G,
    config: &FriConfig<M>,
    inputs: Vec<Vec<Challenge>>,
    challenger: &mut Challenger,
) -> CommitPhaseResult<Challenge, M>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    let mut inputs_iter = inputs.into_iter().peekable();
    let mut folded = inputs_iter.next().unwrap();
    let mut commits = vec![];
    let mut data = vec![];

    while folded.len() > config.blowup() {
        let leaves = RowMajorMatrix::new(folded, config.folding_factor);
        let (commit, prover_data) = config.mmcs.commit_matrix(leaves);
        challenger.observe(commit.clone());

        let beta: Challenge = challenger.sample_ext_element();
        // We passed ownership of `current` to the MMCS, so get a reference to it
        let leaves = config.mmcs.get_matrices(&prover_data).pop().unwrap();
        folded = g.fold_matrix(beta, leaves.as_view(),config.folding_factor);

        commits.push(commit);
        data.push(prover_data);

        if let Some(v) = inputs_iter.next_if(|v| v.len() == folded.len()) {
            izip!(&mut folded, v).for_each(|(c, x)| *c += x);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(folded.len(), config.blowup());
    let final_poly = folded[0];
    for x in folded {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

fn answer_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::ProverData<RowMajorMatrix<F>>],
    index: usize,
) -> Vec<CommitPhaseProofStep<F, M>>
where
    F: Field,
    M: Mmcs<F>,
{
    commit_phase_commits
        .iter() 
        .enumerate()
        .map(|(i, commit)| {
            // todo: apply folding factor 
            println!("i:{:?}", i);
            // println!("log_folding_factor:{:?}", config.log_folding_factor);

            //fix index_i = index / (folding_factor^i)

            // calculate the index for the new polynomial
            let index_i = index >> (config.log_folding_factor * i);
            println!("index_i:{}", index_i);
            // calculate the index for the polynomial-matrix which each row compose with folding_factor evaluation.
            let leaf_index = index_i >> config.log_folding_factor; // >> log_K
            println!("leaf_index:{}", leaf_index);

            // we need to make sure the point open at the correctly position 
            let (mut opened_rows, opening_proof) = config.mmcs.open_batch(leaf_index, commit);

            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            assert_eq!(opened_row.len(), config.folding_factor, "the number of committed data should be euqal to folding_factor");

            // modify the CommitPhaseProofStep
            // let sibling_value = opened_row[index_i_sibling % 2];

            // Notice: we modify the CommitPhaseProofStep here
            CommitPhaseProofStep {
                opened_row,
                opening_proof,
            }
        })
        .collect()
}
