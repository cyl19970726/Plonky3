use p3_dft::TwoAdicSubgroupDft;
use p3_fri::{LdtConfig, LdtParam, SoundnessType};

pub struct StirConfig<M> {
    pub param: LdtParam,
    pub num_rounds: usize,
    // (cur_log_degree,cur_log_blowup,cur_repetition)
    pub rounds_info: Vec<(usize, usize, usize)>,
    pub odd_samples: usize,
    mmcs: M,
}

impl<M> StirConfig<M> {
    fn new(param: LdtParam, m: M) -> Self {
        // compute num_rounds
        let folding_factor = 1 << param.log_folding_factor;

        // d = 8  = 3bit   k =2=1bit  8->4->2->1  3/1 = 3 round
        // d = 16 = 4bit   k =4=2bit  16->4->1    4/2 = 2 round
        let num_rounds = param.log_start_degree / param.log_folding_factor;

        let rounds_info = (0..num_rounds)
            .into_iter()
            .map(|round| {
                // compute degree
                let cur_log_degree =
                    param.log_start_degree - (num_rounds * param.log_folding_factor);

                // compute rates
                let cur_log_blowup = param.log_blowup + round * (param.log_folding_factor - 1);

                // compute repetitions
                let cur_repetition = param.num_queries(cur_log_blowup);
                (cur_log_degree, cur_log_blowup, cur_repetition)
            })
            .collect::<Vec<(usize, usize, usize)>>();

        Self {
            param,
            num_rounds,
            rounds_info,
            odd_samples: 2,
            mmcs: m,
        }
    }
}

impl<M> LdtConfig<M> for StirConfig<M> {
    fn num_queries(&self, log_inv_rate: usize) -> usize {
        let constant = match self.soundness_type() {
            SoundnessType::Provable => 2,
            SoundnessType::Conjecture => 1,
        };
        ((constant * self.protocol_security_level()) as f64 / log_inv_rate as f64).ceil() as usize
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
