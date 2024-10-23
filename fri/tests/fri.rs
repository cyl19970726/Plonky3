use core::cmp::Reverse;
use std::marker::PhantomData;

use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
use p3_challenger::{CanSampleBits, DuplexChallenger, FieldChallenger};
use p3_commit::ExtensionMmcs;
use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::extension::BinomialExtensionField;
use p3_field::{AbstractField, Field};
use p3_fri::{prover, verifier, FriConfig, LdtProver, LdtVerifer, TwoAdicFriGenericConfig};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_matrix::Matrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use p3_util::log2_strict_usize;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use prover::is_power_of_k;
type Val = BabyBear;
type Challenge = BinomialExtensionField<Val, 4>;

type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
type ValMmcs =
    MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
type MyFriConfig = FriConfig<ChallengeMmcs>;

fn get_ldt_for_testing<R: Rng>(rng: &mut R, log_folding_factor: usize) -> (Perm, MyFriConfig) {
    let perm = Perm::new_from_rng_128(
        Poseidon2ExternalMatrixGeneral,
        DiffusionMatrixBabyBear::default(),
        rng,
    );
    let hash = MyHash::new(perm.clone());
    let compress = MyCompress::new(perm.clone());
    let mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
    let fri_config = FriConfig {
        log_blowup: 2,
        num_queries: 10,
        folding_factor: 1 << log_folding_factor,
        log_folding_factor,
        proof_of_work_bits: 8,
        mmcs,
    };
    (perm, fri_config)
}

fn do_test_fri_ldt<R: Rng>(rng: &mut R,log_folding_factor: usize) {
    let folding_factor = 1 << log_folding_factor;
    let (perm, fc) = get_ldt_for_testing(rng,log_folding_factor);
    let dft = Radix2Dit::default();
    let shift = Val::generator();

    let degrees = vec![8,10];
    let ldes: Vec<RowMajorMatrix<Val>> = degrees.iter()
        .map(|deg_bits| {
            let evals = RowMajorMatrix::<Val>::rand_nonzero(rng, 1 << deg_bits, 1);
            println!("evals len:{:?}", (1 << deg_bits));
            //fix added_bit
            let mut lde = dft.coset_lde_batch(evals, fc.log_blowup, shift);
            reverse_matrix_index_bits(&mut lde);
            lde
        })
        .collect();
    ldes.iter().for_each(|it|{
        println!("{:?}",it.height());
    });

    let (proof, p_sample) = {
        // Prover world
        let mut chal = Challenger::new(perm.clone());
        let alpha: Challenge = chal.sample_ext_element();

        let input: [_; 32] = core::array::from_fn(|log_height| {
            let matrices_with_log_height: Vec<&RowMajorMatrix<Val>> = ldes
                .iter()
                .filter(|m| log2_strict_usize(m.height()) == log_height)
                .collect();
            if matrices_with_log_height.is_empty() {
                None
            } else {
                let reduced: Vec<Challenge> = (0..(1 << log_height))
                    .map(|r| {
                        alpha
                            .powers()
                            .zip(matrices_with_log_height.iter().flat_map(|m| m.row(r)))
                            .map(|(alpha_pow, v)| alpha_pow * v)
                            .sum()
                    })
                    .collect();
                Some(reduced)
            }
        });

        let input: Vec<Vec<Challenge>> = input.into_iter().rev().flatten().collect();

        let log_max_height = log2_strict_usize(input[0].len());

        let fri_prover = prover::FriProver::new(&fc);
        let proof = fri_prover.prove(
            &TwoAdicFriGenericConfig::<Vec<(usize, Challenge)>, ()>(PhantomData),
        input.clone(),
        &mut chal,
        |idx| {
                // As our "input opening proof", just pass through the literal reduced openings.
                let mut ro = vec![];
                for v in &input {
                    //TODO:
                    let log_height = log2_strict_usize(v.len());
                    //fix
                    ro.push((log_height, v[idx >> (log_folding_factor * (log_max_height - log_height))]));
                }
                ro.sort_by_key(|(lh, _)| Reverse(*lh));
                ro
            },
        );
        // let proof = prover::prove(
        //     &TwoAdicFriGenericConfig::<Vec<(usize, Challenge)>, ()>(PhantomData),
        //     &fc,
        //     input.clone(),
        //     &mut chal,
        //     |idx| {
        //         // As our "input opening proof", just pass through the literal reduced openings.
        //         let mut ro = vec![];
        //         for v in &input {
        //             let log_height = log2_strict_usize(v.len());
        //             ro.push((log_height, v[idx >> (log_max_height - log_height)]));
        //         }
        //         ro.sort_by_key(|(lh, _)| Reverse(*lh));
        //         ro
        //     },
        // );

        (proof, chal.sample_bits(8))
    };

    let mut v_challenger = Challenger::new(perm);
    let _alpha: Challenge = v_challenger.sample_ext_element();

    let verifier = verifier::Verifier::new(&fc);
    verifier.verify(
        &TwoAdicFriGenericConfig::<Vec<(usize, Challenge)>, ()>(PhantomData),
        &proof,
        &mut v_challenger,
        |_index, proof| Ok(proof.clone()),
    )
    .unwrap();

    assert_eq!(
        p_sample,
        v_challenger.sample_bits(8),
        "prover and verifier transcript have same state after FRI"
    );
}

#[test]
fn test_fri_ldt() {
    tracing_subscriber::fmt::init();
    tracing::info!("开始 FRI LDT 测试");
    // FRI is kind of flaky depending on indexing luck
    // for i in 0..4 {
    let mut rng = ChaCha20Rng::seed_from_u64(0);
    do_test_fri_ldt(&mut rng,1);
    // }
}


#[test]
fn test_fri_ldt_k() {
    tracing_subscriber::fmt::init();
    tracing::info!("开始 FRI LDT 测试");
    // FRI is kind of flaky depending on indexing luck
    for i in 0..4 {
        let mut rng = ChaCha20Rng::seed_from_u64(i);
        do_test_fri_ldt(&mut rng,2);
    }
}
