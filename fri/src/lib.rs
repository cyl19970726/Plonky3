//! An implementation of the FRI low-degree test (LDT).

#![no_std]

extern crate alloc;

mod config;
mod fold_even_odd;
mod proof;
pub mod prover;
mod two_adic_pcs;
pub mod verifier;

use alloc::vec::Vec;
pub use config::*;
pub use fold_even_odd::*;
pub use proof::*;
// use rand_chacha::rand_core::impls;
use serde::de::Error;
pub use two_adic_pcs::*;
use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};
use verifier::FriError;

// todo: config 
pub trait LdtProver<'a, G, Val, Challenge, M, Challenger> 
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>
{
	type Proof;

    fn new(g: &'a FriConfig<M>) -> Self;

    fn folding_factor(&self) -> usize;

    fn prove(&self,   
        g: &G,
        inputs: Vec<Vec<Challenge>>,
        challenger: &mut Challenger,
        open_input: impl Fn(usize) -> G::InputProof
    ) -> Self::Proof;    
}

pub trait LdtVerifer<'a,G, Val, Challenge, M, Challenger>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>
{

    type Proof;
    fn new(config: &'a FriConfig<M>) -> Self;

    fn folding_factor(&self) -> usize;

    fn verify(
        &self,
        g: &G,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        open_input: impl Fn(usize, &G::InputProof) -> Result<Vec<(usize, Challenge)>, G::InputError>,
    ) -> Result<(),FriError<M::Error, G::InputError>>;
}