//! An implementation of the FRI low-degree test (LDT).

// #![no_std]

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
use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_matrix::dense::RowMajorMatrix;
use p3_util::log2_strict_usize;
use serde::de::Error;
use tracing::{info_span, instrument};
pub use two_adic_pcs::*;
use verifier::FriError;

pub trait LdtProver<'a, G, Val, Challenge, M, Challenger>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    type Proof;
    type Conf: LdtConfig<M>;

    fn new(g: &'a Self::Conf) -> Self;

    fn prove(
        &self,
        g: &G,
        inputs: Vec<Vec<Challenge>>,
        challenger: &mut Challenger,
        open_input: impl Fn(usize) -> G::InputProof,
    ) -> Self::Proof;
}

pub trait LdtVerifer<'a, G, Val, Challenge, M, Challenger>
where
    Val: Field,
    Challenge: ExtensionField<Val>,
    M: Mmcs<Challenge>,
    Challenger: FieldChallenger<Val> + GrindingChallenger + CanObserve<M::Commitment>,
    G: FriGenericConfig<Challenge>,
{
    type Proof;
    type Conf;
    fn new(config: &'a Self::Conf) -> Self;

    fn verify(
        &self,
        g: &G,
        proof: &Self::Proof,
        challenger: &mut Challenger,
        open_input: impl Fn(usize, &G::InputProof) -> Result<Vec<(usize, Challenge)>, G::InputError>,
    ) -> Result<(), LdtError<M::Error, G::InputError>>;
}

pub trait LdtConfig<M> {

    fn new_without_secbits(log_folding_factor: usize, query_num: usize, log_blowup: usize,  pow_bits: usize, m: M) -> Self;

    fn new_without_querynum(log_folding_factor: usize, query_num: usize, log_blowup: usize,  pow_bits: usize, m: M) -> Self;

    fn num_queries(&self, log_inv_rate: usize) -> usize;

    fn log_folding_factor(&self) -> usize;

    fn pow_bits(&self) -> usize;

    fn log_blowup(&self) -> usize;

    fn blowup(&self) -> usize {
        1 << self.log_blowup()
    }

    fn protocol_security_level(&self) -> usize;

    fn soundness_type(&self) -> SoundnessType;

    fn get_mmcs(&self) -> &M;
}

#[derive(Debug)]
pub struct LdtParam {
    // log_rho_inv
    pub log_blowup: usize,
    pub num_queries: usize,
    pub log_folding_factor: usize,
    pub pow_bits: usize,
    pub protocol_security_level: usize,
    pub soundness_type: SoundnessType,
}

impl LdtParam {
    pub fn new_without_secbits(log_folding_factor: usize, query_num: usize, log_blowup: usize,  pow_bits: usize) -> Self{
        let sec_bits = protocol_security_level(log_blowup, SoundnessType::Conjecture, query_num);
        Self{
            log_blowup,
            num_queries: query_num,
            log_folding_factor,
            pow_bits,
            protocol_security_level: sec_bits + pow_bits,
            soundness_type: SoundnessType::Conjecture,
        }
    }

    pub fn new_without_querynum(log_folding_factor: usize, sec_bits: usize, log_blowup: usize,  pow_bits: usize) -> Self{
        let query_num = num_queries(log_blowup, SoundnessType::Conjecture, sec_bits - pow_bits);
        Self{
            log_blowup,
            num_queries: query_num,
            log_folding_factor,
            pow_bits,
            protocol_security_level: sec_bits,
            soundness_type: SoundnessType::Conjecture,
        }
    }

    pub fn num_queries(&self, log_inv_rate: usize) -> usize {
        let constant = match self.soundness_type {
            SoundnessType::Provable => 2,
            SoundnessType::Conjecture => 1,
        };
        ((constant * self.protocol_security_level) as f64 / log_inv_rate as f64).ceil() as usize
    }

}

pub fn num_queries(log_inv_rate: usize, soundness_type: SoundnessType, protocol_security_level: usize) -> usize {
    let constant = match soundness_type {
        SoundnessType::Provable => 2,
        SoundnessType::Conjecture => 1,
    };
    ((constant * protocol_security_level) as f64 / log_inv_rate as f64).ceil() as usize
}

pub fn protocol_security_level(log_inv_rate: usize, soundness_type: SoundnessType, num_queries: usize) -> usize {
    let constant = match soundness_type {
        SoundnessType::Provable => 2,
        SoundnessType::Conjecture => 1,
    };
    ((num_queries as f64 * log_inv_rate as f64) / constant as f64).ceil() as usize
}



#[derive(Debug, Clone, Copy)]
pub enum SoundnessType {
    Provable,
    Conjecture,
}

#[derive(Debug)]
pub enum StirError<CommitMmcsErr, InputError> {
    InvalidShake,
    CommitPhaseMmcsError(CommitMmcsErr),
    InputError(InputError),
    FinalPolyMismatch,
    FinalPolyCheckErr,
    InvalidPowWitness,
}

#[derive(Debug)]
pub enum LdtError<CommitMmcsErr, InputError> {
    Stir(StirError<CommitMmcsErr, InputError>),
    Fri(FriError<CommitMmcsErr, InputError>),
}

impl<CommitMmcsErr, InputError> From<FriError<CommitMmcsErr, InputError>>
    for LdtError<CommitMmcsErr, InputError>
{
    fn from(value: FriError<CommitMmcsErr, InputError>) -> Self {
        LdtError::Fri(value)
    }
}

impl<CommitMmcsErr, InputError> From<StirError<CommitMmcsErr, InputError>>
    for LdtError<CommitMmcsErr, InputError>
{
    fn from(value: StirError<CommitMmcsErr, InputError>) -> Self {
        LdtError::Stir(value)
    }
}
