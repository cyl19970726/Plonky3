use alloc::vec::Vec;

use bitcoin::taproot::{LeafNode, TapLeaf, TaprootMerkleBranch};
use p3_commit::Mmcs;
use p3_field::Field;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Witness: Serialize",
    deserialize = "Witness: Deserialize<'de>"
))]
pub struct FriProof<F: Field, M: Mmcs<F>, Witness> {
    pub(crate) commit_phase_commits: Vec<M::Commitment>,
    pub(crate) query_proofs: Vec<BfQueryProof>,
    // This could become Vec<FC::Challenge> if this library was generalized to support non-constant
    // final polynomials.
    pub(crate) final_poly: F,
    pub(crate) pow_witness: Witness,
}

#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BfQueryProof {
    /// For each commit phase commitment, this contains openings of a commit phase codeword at the
    /// queried location, along with an opening proof.
    pub(crate) commit_phase_openings: Vec<BfCommitPhaseProofStep>,
}

#[derive(Serialize, Deserialize)]
// #[serde(bound(serialize = "F: Serialize"))]
#[serde(bound = "")]
pub struct BfCommitPhaseProofStep {
    /// The opening of the commit phase codeword at the sibling location.
    // This may change to Vec<FC::Challenge> if the library is generalized to support other FRI
    // folding arities besides 2, meaning that there can be multiple siblings.
    pub(crate) leaf_node: TapLeaf,

    pub(crate) merkle_branch: TaprootMerkleBranch,
}

pub fn get_leaf_index_by_query_index(query_index: usize) -> (usize, usize, usize) {
    let index_i = query_index >> 1;
    let index_i_sibling = index_i ^ 1;
    let index_pair = index_i >> 1;
    (index_pair, index_i, index_i_sibling)
}