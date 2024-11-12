use alloc::vec::Vec;
use core::fmt::Debug;

use p3_field::Field;
use p3_matrix::Matrix;

use crate::{LdtConfig, LdtParam, SoundnessType};

#[derive(Debug)]
pub struct FriConfig<M> {
    pub param: LdtParam,
    pub mmcs: M,
}


impl<M> LdtConfig<M> for FriConfig<M> {

    fn new_without_querynum(log_folding_factor: usize, sec_bits: usize, log_blowup: usize,  pow_bits: usize, m : M) -> Self {
        Self{
            param: LdtParam::new_without_querynum(log_folding_factor, sec_bits, log_blowup, pow_bits),
            mmcs: m,
        }
    }

    fn new_without_secbits(log_folding_factor: usize, query_num: usize, log_blowup: usize,  pow_bits: usize, m: M) -> Self {
        Self{
            param: LdtParam::new_without_secbits(log_folding_factor, query_num, log_blowup, pow_bits),
            mmcs: m,
        }
    }

    fn num_queries(&self, _log_inv_rate: usize) -> usize {
        self.param.num_queries
    }

    fn log_folding_factor(&self) -> usize {
        self.param.log_folding_factor
    }

    fn log_blowup(&self) -> usize {
        self.param.log_blowup
    }

    fn pow_bits(&self) -> usize {
        self.param.pow_bits
    }

    fn protocol_security_level(&self) -> usize {
        self.param.protocol_security_level
    }

    fn soundness_type(&self) -> SoundnessType {
        self.param.soundness_type
    }

    fn get_mmcs(&self) -> &M {
        &self.mmcs
    }
}


/// Whereas `FriConfig` encompasses parameters the end user can set, `FriGenericConfig` is
/// set by the PCS calling FRI, and abstracts over implementation details of the PCS.
pub trait FriGenericConfig<F: Field> {
    type InputProof;
    type InputError: Debug;

    /// We can ask FRI to sample extra query bits (LSB) for our own purposes.
    /// They will be passed to our callbacks, but ignored (shifted off) by FRI.
    fn extra_query_index_bits(&self) -> usize;

    /// Fold a row, returning a single column.
    /// Right now the input row will always be 2 columns wide,
    /// but we may support higher folding arity in the future.
    fn fold_row(
        &self,
        index: usize,
        log_height: usize,
        beta: F,
        evals: impl Iterator<Item = F>,
        folding_factor: usize,
    ) -> F;

    /// Same as applying fold_row to every row, possibly faster.
    fn fold_matrix<M: Matrix<F>>(&self, beta: F, m: M, folding_factor: usize) -> Vec<F>;
}
