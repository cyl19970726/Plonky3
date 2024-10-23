use p3_field::{eval_poly, Field, PrimeField, PrimeField32, TwoAdicField};
use p3_util::{log2_strict_usize, reverse_slice_index_bits};

use core::fmt::{Debug, Display};
use core::hash::Hash;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug,Clone,PartialEq, Eq)]
pub enum PolyType {
    Coefficient,
    Evaluation,
}

#[derive(Debug,Clone,PartialEq, Eq)]
struct Polynomial<F: PrimeField32 + TwoAdicField>{
    values: Vec<F>,
    points: Option<Vec<F>>,
    ty: PolyType,
}

impl <F: PrimeField32 + TwoAdicField> Polynomial<F> {
    fn new(values: Vec<F>,ty: PolyType,points: Option<Vec<F>>) -> Self{
        Self{
            values,
            points,
            ty,
        }
    }

    fn evaluate(&self,point: F) -> F {
        assert!(self.ty == PolyType::Coefficient);
        self.values.iter().enumerate().fold(F::zero(), |acc,item|{
            acc + point.exp_u64(item.0 as u64)  * *item.1
        })
    }

    /// p(x) = \sum_{i=0}{i=n}y_iL_i(X)
    /// 
    /// L_i(x) = \frac{(x - x_0)(x - x_1) \cdots (x - x_{i-1})(x - x_{i+1}) \cdots (x - x_n)}{(x_i - x_0)(x_i - x_1) \cdots (x_i - x_{i-1})(x_i - x_{i+1}) \cdots (x_i - x_n)}

    // Interpolation function
    fn interpolation(&self, interpolate_point: F) -> u32 {
        assert_eq!(self.ty, PolyType::Evaluation);
        
        // Summing the Lagrange basis contributions for each value
        self.values.iter().enumerate().fold(0, |acc, (index, eval)| {
            acc + (*eval).as_canonical_u32() * Self::lagrange_basis(index, &self.values, interpolate_point)
        })
    }

    // Lagrange basis function
    fn lagrange_basis(index: usize, x_values: &[F], interpolate_point: F) -> u32 {
        x_values.iter().enumerate().fold(1, |acc, (i, &x_val)| {
            if i != index {
                acc * (interpolate_point.as_canonical_u32() - x_val.as_canonical_u32()) / (x_values[index].as_canonical_u32() - x_val.as_canonical_u32())
            } else {
                acc
            }
        })
    }

    /// p(x) = even(x^2) + x odd(x^2)
    /// p(-x) = even(x^2) - x odd(x^2) 
    fn fft(&self) -> Self{
        assert!(self.ty== PolyType::Coefficient);
        if self.values.len() == 1{
            return self.clone();
        }
        let bits_len = log2_strict_usize(self.values.len());
      
        let cur_domain_generator = F::two_adic_generator(bits_len);
        let mut cur_domain = vec![];
        for i in 0..self.values.len(){
            cur_domain.push(cur_domain_generator.exp_u64(i as u64));
        }
        let even = Self::new(self.values.clone().into_iter().step_by(2).collect(), self.ty.clone(), None).fft();
        let odd = Self::new(self.values.clone().into_iter().skip(1).step_by(2).collect(), self.ty.clone(), None).fft();
        
        // I will compose this function now 
        // now I maybe have the evaluation of even(x^2) and the odd(x^2) , but how to compose them 
        // compose the evaluation 
        reverse_slice_index_bits(&mut cur_domain);
        let mut evals = vec![];

        cur_domain.iter().step_by(2).zip(cur_domain.iter().skip(1).step_by(2)).enumerate().for_each(|(index,(x,neg_x))|{
            assert_eq!(*x + (*neg_x),F::zero());
            evals.push(even.values[index] + odd.values[index]**x);
            evals.push(even.values[index] - odd.values[index]**x);
        });

        Self::new(evals, PolyType::Evaluation, Some(cur_domain))
    }    

    
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{thread_rng, Rng};
    use p3_field::{AbstractField,Field};
    use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};

    use  p3_baby_bear::BabyBear;
    #[test]
    fn test_interpolation(){
        type F = BabyBear;

        let mut rng = thread_rng();

        let log_n = 2;
        let n = 1 << log_n;
        // let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let coeff = vec![F::one(),F::one(),F::one()];// 1 + x + x^2 

        let evaluation = vec![F::one(),F::from_canonical_u32(3),F::from_canonical_u32(7)];
        let domain = vec![F::zero(),F::from_canonical_u32(1),F::from_canonical_u32(2)];

        // lagerange basis check
        let v1 = Polynomial::<BabyBear>::lagrange_basis(0, &domain, F::from_canonical_u32(0));
        assert_eq!(v1,1);
        let v1 = Polynomial::<BabyBear>::lagrange_basis(1, &domain, F::from_canonical_u32(1));
        assert_eq!(v1,1);
        let v1 = Polynomial::<BabyBear>::lagrange_basis(2, &domain, F::from_canonical_u32(2));
        assert_eq!(v1,1);

        let eval_poly = Polynomial::<F>::new(evaluation, PolyType::Evaluation, Some(domain));
        let interpolation_value = eval_poly.interpolation(F::from_canonical_u32(2));
        
        let coeff_poly = Polynomial::<F>::new(coeff, PolyType::Coefficient, None);
        let eval_value = coeff_poly.evaluate(F::from_canonical_u32(2));
        assert_eq!(interpolation_value,eval_value.as_canonical_u32());

    }


    #[test]
    fn test_fft(){
        type F = BabyBear;

        let mut rng = thread_rng();

        let coeff = vec![F::one(),F::one(),F::one(),F::one()];// 1 + x + x^2  + x^3
        let coeff_poly = Polynomial::new(coeff.clone(), PolyType::Coefficient, None);
        let eval_poly = coeff_poly.fft();
        println!("{:?}",eval_poly);

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeff.clone());
        assert_eq!(evals,eval_poly.values);
    }
    
    #[test]
    fn test_fft_rnd(){
        type F = BabyBear;
        let mut rng = thread_rng();
        let log_n = 8;
        let n = 1 << log_n;
        let coeffs = (0..n).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let dft = Radix2Dit::default();
        let evals = dft.dft(coeffs.clone());

        let coeffs_poly = Polynomial::new(coeffs.clone(), PolyType::Coefficient, None);
        let mut eval_poly = coeffs_poly.fft();
        let mut act_evals = eval_poly.values;
        reverse_slice_index_bits(&mut act_evals);
        assert_eq!(evals,act_evals);
    }
}
