use alloc::vec;
use alloc::vec::Vec;
use core::iter;

use itertools::{izip, Itertools};
use p3_air::Air;
use p3_challenger::{CanObserve, CanSample, FieldChallenger};
use p3_commit::{Pcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, AbstractField, PackedValue};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;
use tracing::{info_span, instrument};

use crate::symbolic_builder::{get_log_quotient_degree, SymbolicAirBuilder};
use crate::{
    Commitments, Domain, OpenedValues, PackedChallenge, PackedVal, Proof, ProverConstraintFolder,
    StarkGenericConfig, Val,
};

#[instrument(skip_all)]
#[allow(clippy::multiple_bound_locations)] // cfg not supported in where clauses?
pub fn prove<
    SC,
    #[cfg(debug_assertions)] A: for<'a> Air<crate::check_constraints::DebugConstraintBuilder<'a, Val<SC>>>,
    #[cfg(not(debug_assertions))] A,
>(
    config: &SC,
    air: &A,
    challenger: &mut SC::Challenger,
    trace: RowMajorMatrix<Val<SC>>,
    public_values: &Vec<Val<SC>>,
) -> Proof<SC>
where
    SC: StarkGenericConfig,
    A: Air<SymbolicAirBuilder<Val<SC>>> + for<'a> Air<ProverConstraintFolder<'a, SC>>,
{

    /*  

    a complete process of uni-stark  
    assume prover generate sa proof for the below fib-trace and then verifier verifies this proof.
    fib trace 
    a b   column 
    1 2   input0 input1 
    2 3
    3 5
    5 8
    8 13   output 
    
    1. commit trace(X) and get the commitment trace_root
         lde_trace(X) = lde(trace(X)) 
         commit lde_trace(X) using Merkletree Commitment and get the trace_commit
    2. generate quotient poly at coset
         1 == input0  2==input1              when_first_row 
         13 == output                        when_last_rot
         C_0(X) = a(X) + b(X) - b(gX)        when_transition
         C_1(X) = a(gX) - b(X)               when_transition
         q_0(X) = C_0(X) / Z_H(X)         
         q_1(X) = C_1(X) / Z_H(X)
         degree adjust with alpha and get q_0'(X) and q_1'(X)
         Q(X) = q_0'(X) + q_1'(X)  domain at coset
    3. split Q_(X) as Q_1(X) and Q_2(X) through Q(X) = Q_1(X^2) + X*Q_2(X^2)
    4. commit Q_1(X), Q_2(X) and get the commitment quotient_commit
         4.1 Low Degree Extension at coset   
             lde_Q_1(X) = lde(Q_1(X))
             lde_Q_2(X) = lde(Q_2(X))
         4.2 Merkle Tree Commitment  
             places the lde_Q_1(X) and lde_Q_2(X) into a matrix like below sequence and commit the matrix
             { lde_Q_1(X)_matrix row 0, lde_Q_2(X)_matrix row 0  }
             { lde_Q_1(X)_matrix row 1, lde_Q_2(X)_matrix row 1  }
             {                     ......                        }
             { lde_Q_1(X)_matrix row n, lde_Q_2(X)_matrix row n  }
             commit this matrix using Merkletree Commitment and get the quotient_commit
    5. open trace(X) at zeta,zeta_next and open Q_1(X),Q_2(X) at zeta
       5.1 compute reduce polynomial 
             ldt_0(X) = (trace(X) - trace(zeta)) / (X - zeta)         
             ldt_0'(X) = (trace(X) - trace(zeta_next)) / (X - zeta_next)
             compute reduce polynomail reduce_0(X):
             reduce_0(X) = (alpha^0 * ldt_0(X)_0 ...+  alpha^i *ldt_0(X)_i) +(alpha^i+1) * ldt_0'(X) + (alpha^i+j)* ldt_0'(X)_j   【i is the width of ldt_0(X) matrix，j is the width of ldt_0'(X) matrix】 
             low degree test for reduce_0(X):
    
             ldt_2(X) = Q_1(X) - Q_1(zeta) / (X - zeta)
             ldt_3(X) = Q_2(X) - Q_2(zeta) / (X - zeta)
             compute reduce polynomial reduce_1(X):
             reduce_1(X) = (alpha^0 * ldt_2(X)_0 ...  alpha^i *ldt_2(X)_i) +(alpha^i+1) *...* ldt_3'(X) + (alpha^i+j+k+l)* ldt_2'(X)_j = 0  【i,j,k,l is the width of ldt_2(X),ldt_2'(X),ldt_3(X),ldt_3'(X)】 
        
        5.2 low degree test for reduce polynomial
             low_degree_test for vec![reduce_0(X),reduce_1(X)]
        
        5.3 output
        reduce_0(X) output:
             open_values: 
                 trace(zeta), trace(zeta_next)
             fri_input proof: 
                 fri sample challenge point: beta
                 trace(beta) + merkle_path
                 trace(-beta) + merkle_path
         reduce_1(X) output:
             open_values: 
                 Q_1(zeta), Q_2(zeta),
             fri_input proof:
                 fri sample challenge point: beta
                 Q_1(beta) + merkle_path
                 Q_2(beta) + merkle_path
                 Q_1(-beta) + merkle_path
                 Q_2(-beta) + merkle_path
     
          low_degree_test output:
             low_degree_test proof
    6. output stark proof
    
    ======= Query Phase ========
    
    1. verify the open-values trace(zeta),trace(zeta_next),Q_1(zeta),Q_2(zeta)
       so the input of this partion is:
       ***********************************************
       * reduce random sample values: alpha          *
       * open values: trace(zeta)  trace(zeta_next)  *
       * open values: trace(zeta) trace(zeta_next)   *
       * fri challenge point: beta                   * 
       * merkle path: trace_root MPT(trace(beta))  MPT(trace(-beta)) *
       * merkle path: quotient_root MPT(Q_1(beta))   MPT(Q_1(-beta)) MPT(Q_2(beta))   MPT(Q_1(-beta))*
       * fri_low_degree_test proof                   *
       ***********************************************
       1.1 verify trace(zeta),trace(zeta_next) is the trace(X) evaluates at zeta and zeta_next point
            ldt_0(beta) = trace(beta) - trace(zeta) / (beat - zeta)      
            ldt_0'(beta) = trace(beta) - trace(zeta_next) / (beta - zeta_next)
            reduce_0(beta) = (alpha^0 * ldt_0(beta)_0 ...+  alpha^i *ldt_0(beta)_i) +(alpha^i+1) * ldt_0'(beta) + (alpha^i+j)* ldt_0'(beta)_j
       1.2 verify Q_1(zeta),Q_2(zeta) is the Q_1(X) and Q_2(X) evaluates at zeta point
            ldt_2(beta) = Q_1(beta) - Q_1(zeta) / (beta - zeta)
            ldt_3(beta) = Q_2(beta) - Q_2(zeta) / (beta - zeta)
            reduce_1(beta) = (alpha^0 * ldt_2(beta)_0 ...  alpha^i *ldt_2(beta)_i) +(alpha^i+1) *...* ldt_3'(beta) + (alpha^i+j+k+l)* ldt_2'(beta)_j
       1.3 verify low_degree_test_proof for vec![reduce_0(beta) and reduce_1(beta)]
       1.4 we need to notice that a compelete FRI Query phase is component with multiple query, 
           and for each query the verifier samples a new point so-called beta to challenge the 
           commited poly which leads to the steps of 1.1,1.2,1.3 is repeated for each challenge
    2. evaluate the committed Q(X) at zeta
        calculate zps
        calculate quotient
    3. compute Q(zeta) by trace(zeta), trace(zeta_next)
    4. verify the relationship between Q(zeta) and Q_1(zeta),Q_2(zeta)
    */

    #[cfg(debug_assertions)]
    crate::check_constraints::check_constraints(air, &trace, public_values);

    let degree = trace.height();
    let log_degree = log2_strict_usize(degree);

    let log_quotient_degree = get_log_quotient_degree::<Val<SC>, A>(air, 0, public_values.len());
    let quotient_degree = 1 << log_quotient_degree;

    let pcs = config.pcs();
    // shift = 1 
    let trace_domain = pcs.natural_domain_for_degree(degree);

    // PCS::commmit the trace
    // Question: each column is a polynomial how to commit?
    let (trace_commit, trace_data) =
        info_span!("commit to trace data").in_scope(|| pcs.commit(vec![(trace_domain, trace)]));//degree 9 -> degree 10 w=two_adbic(10 bits)

    // Observe the instance.
    challenger.observe(Val::<SC>::from_canonical_usize(log_degree));
    // TODO: Might be best practice to include other instance data here; see verifier comment.

    challenger.observe(trace_commit.clone());
    challenger.observe_slice(public_values);
    let alpha: SC::Challenge = challenger.sample_ext_element();

    // shift*generator = 0x1f 
    let quotient_domain =
        trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));

    let trace_on_quotient_domain = pcs.get_evaluations_on_domain(&trace_data, 0, quotient_domain);

    let quotient_values = quotient_values(
        air,
        public_values,
        trace_domain,
        quotient_domain,
        trace_on_quotient_domain,
        alpha,
    );
    let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();

    // let quotient_chunk_size = 1 << (log_degree + log_quotient_degree) / 1 << log_quotient_degree = 1 << log_degree
    // let quotient_degree = 1 << log_quotient_degree;
    // Question: Why need to split domain and evals?
    let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);// 隐含了一个调整degree的过程？
    
    // 1,2,3,4,5,6,7,8,9,10,11,12 M - evaluation at [1..12]
    // domain F_13 (generator 2) 
    // S = 3 
    // 1, 4, 7, 10 M1   [5,12,8,1]domain F_13
    // 2, 5, 8, 11 M2   [10,11,3,2] domain F_13
    // 3, 6, 9, 12 M3   [7,9,6,3] domain F_13
    let qc_domains = quotient_domain.split_domains(quotient_degree);

    let (quotient_commit, quotient_data) = info_span!("commit to quotient poly chunks")
        .in_scope(|| pcs.commit(izip!(qc_domains, quotient_chunks).collect_vec()));
    challenger.observe(quotient_commit.clone());

    let commitments = Commitments {
        trace: trace_commit,
        quotient_chunks: quotient_commit,
    };

    let zeta: SC::Challenge = challenger.sample();
    let zeta_next = trace_domain.next_point(zeta).unwrap();

    let (opened_values, opening_proof) = pcs.open(
        vec![
            (&trace_data, vec![vec![zeta, zeta_next]]), // why the first element different the second element?
            (
                &quotient_data,
                // open every chunk at zeta
                (0..quotient_degree).map(|_| vec![zeta]).collect_vec(),
            ),
        ],
        challenger,
    );
    let trace_local = opened_values[0][0][0].clone();//[round][matrix][point] = ys
    let trace_next = opened_values[0][0][1].clone();
    let quotient_chunks = opened_values[1].iter().map(|v| v[0].clone()).collect_vec(); // v = matrix, v[0] = v[point] is also ys
    let opened_values = OpenedValues {
        trace_local,
        trace_next,
        quotient_chunks,
    };
    Proof {
        commitments,
        opened_values,
        opening_proof,
        degree_bits: log_degree,
    }
}

#[instrument(name = "compute quotient polynomial", skip_all)]
fn quotient_values<SC, A, Mat>(
    air: &A,
    public_values: &Vec<Val<SC>>,
    trace_domain: Domain<SC>,
    quotient_domain: Domain<SC>,
    trace_on_quotient_domain: Mat,
    alpha: SC::Challenge,
) -> Vec<SC::Challenge>
where
    SC: StarkGenericConfig,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    Mat: Matrix<Val<SC>> + Sync,
{
    let quotient_size = quotient_domain.size();
    let width = trace_on_quotient_domain.width();
    let mut sels = trace_domain.selectors_on_coset(quotient_domain);

    let qdb = log2_strict_usize(quotient_domain.size()) - log2_strict_usize(trace_domain.size());
    let next_step = 1 << qdb;

    // We take PackedVal::<SC>::WIDTH worth of values at a time from a quotient_size slice, so we need to
    // pad with default values in the case where quotient_size is smaller than PackedVal::<SC>::WIDTH.
    for _ in quotient_size..PackedVal::<SC>::WIDTH {
        sels.is_first_row.push(Val::<SC>::default());
        sels.is_last_row.push(Val::<SC>::default());
        sels.is_transition.push(Val::<SC>::default());
        sels.inv_zeroifier.push(Val::<SC>::default());
    }

    (0..quotient_size)
        .into_par_iter()
        .step_by(PackedVal::<SC>::WIDTH)
        .flat_map_iter(|i_start| {
            let i_range = i_start..i_start + PackedVal::<SC>::WIDTH;

            let is_first_row = *PackedVal::<SC>::from_slice(&sels.is_first_row[i_range.clone()]);
            let is_last_row = *PackedVal::<SC>::from_slice(&sels.is_last_row[i_range.clone()]);
            let is_transition = *PackedVal::<SC>::from_slice(&sels.is_transition[i_range.clone()]);
            let inv_zeroifier = *PackedVal::<SC>::from_slice(&sels.inv_zeroifier[i_range.clone()]);

            let main = RowMajorMatrix::new(
                iter::empty()
                    .chain(trace_on_quotient_domain.vertically_packed_row(i_start))
                    .chain(trace_on_quotient_domain.vertically_packed_row(i_start + next_step))
                    .collect_vec(),
                width,
            );

            let accumulator = PackedChallenge::<SC>::zero();
            let mut folder = ProverConstraintFolder {
                main,
                public_values,
                is_first_row,
                is_last_row,
                is_transition,
                alpha,
                accumulator,
            };
            air.eval(&mut folder);//make sure all the constraints are satisfied

            // quotient(x) = constraints(x) / Z_H(x)
            let quotient = folder.accumulator * inv_zeroifier;

            // "Transpose" D packed base coefficients into WIDTH scalar extension coefficients.
            (0..core::cmp::min(quotient_size, PackedVal::<SC>::WIDTH)).map(move |idx_in_packing| {
                let quotient_value = (0..<SC::Challenge as AbstractExtensionField<Val<SC>>>::D)
                    .map(|coeff_idx| quotient.as_base_slice()[coeff_idx].as_slice()[idx_in_packing])
                    .collect::<Vec<_>>();
                SC::Challenge::from_base_slice(&quotient_value)
            })
        })
        .collect()
}
