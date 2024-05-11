use std::usize;

use bitcoin::ScriptBuf as Script;
use bitcoin_script::{define_pushable, script};

use super::bitcom::*;
use crate::{
    BCAssignment, BfBaseField, BfExtensionField, BitCommitExtension, BitsCommitment,
    ExtensionBCAssignment,
};
define_pushable!();

pub struct ExtensionPointsLeaf<F: BfBaseField, EF: BfExtensionField<F>> {
    leaf_index_1: usize,
    leaf_index_2: usize,
    points: ExtensionPoints<F, EF>,
}

impl<F: BfBaseField, EF: BfExtensionField<F>> ExtensionPointsLeaf<F, EF> {
    pub fn new_from_assign(
        leaf_index_1: usize,
        leaf_index_2: usize,
        x: EF,
        y: EF,
        x2: EF,
        y2: EF,
        bc_assign: &mut ExtensionBCAssignment<F, EF>,
    ) -> Self {
        let points = ExtensionPoints::<F, EF>::new_from_assign(x, y, x2, y2, bc_assign);
        Self {
            leaf_index_1,
            leaf_index_2,
            points,
        }
    }

    pub fn new(leaf_index_1: usize, leaf_index_2: usize, x: EF, y: EF, x2: EF, y2: EF) -> Self {
        let points = ExtensionPoints::<F, EF>::new(x, y, x2, y2);
        Self {
            leaf_index_1,
            leaf_index_2,
            points,
        }
    }

    pub fn recover_points_euqal_to_commited_point(&self) -> Script {
        let scripts = script! {
            {self.points.p1.recover_point_euqal_to_commited_point()}
            {self.points.p2.recover_point_euqal_to_commited_point()}
            OP_1
        };
        scripts
    }

    pub fn signature(&self) -> Vec<Vec<u8>> {
        let mut p1_sigs = self.points.p1.signature();
        let mut p2_sigs = self.points.p2.signature();
        p2_sigs.append(p1_sigs.as_mut());
        p2_sigs
    }
}

pub struct PointsLeaf<F: BfBaseField> {
    leaf_index_1: usize,
    leaf_index_2: usize,
    points: Points<F>,
}

impl<F: BfBaseField> PointsLeaf<F> {
    pub fn new(
        leaf_index_1: usize,
        leaf_index_2: usize,
        x: F,
        y: F,
        x2: F,
        y2: F,
    ) -> PointsLeaf<F> {
        let points = Points::<F>::new(x, y, x2, y2);
        Self {
            leaf_index_1,
            leaf_index_2,
            points,
        }
    }

    pub fn recover_points_euqal_to_commited_point(&self) -> Script {
        let scripts = script! {
            {self.points.p1.recover_point_euqal_to_commited_point()}
            {self.points.p2.recover_point_euqal_to_commited_point()}
            OP_1
        };
        scripts
    }

    pub fn signature(&self) -> Vec<Vec<u8>> {
        let mut p1_sigs = self.points.p1.signature();
        let mut p2_sigs = self.points.p2.signature();
        p2_sigs.append(p1_sigs.as_mut());
        p2_sigs
    }
}

pub struct ExtensionPoints<F: BfBaseField, EF: BfExtensionField<F>> {
    p1: ExtensionPoint<F, EF>,
    p2: ExtensionPoint<F, EF>,
}

impl<F: BfBaseField, EF: BfExtensionField<F>> ExtensionPoints<F, EF> {
    pub fn new_from_assign(
        x1: EF,
        y1: EF,
        x2: EF,
        y2: EF,
        bc_assign: &mut ExtensionBCAssignment<F, EF>,
    ) -> ExtensionPoints<F, EF> {
        let p1 = ExtensionPoint::<F, EF>::new_from_assign(x1, y1, bc_assign);
        let p2 = ExtensionPoint::<F, EF>::new_from_assign(x2, y2, bc_assign);
        Self { p1, p2 }
    }

    pub fn new(x1: EF, y1: EF, x2: EF, y2: EF) -> ExtensionPoints<F, EF> {
        let p1 = ExtensionPoint::<F, EF>::new(x1, y1);
        let p2 = ExtensionPoint::<F, EF>::new(x2, y2);
        Self { p1, p2 }
    }

    pub fn recover_points_euqal_to_commited_points(&self) -> Script {
        let scripts = script! {
            {self.p1.recover_point_euqal_to_commited_point()}
            {self.p2.recover_point_euqal_to_commited_point()}
        };
        scripts
    }

    pub fn signature(&self) -> Vec<Vec<u8>> {
        let mut p1_sigs = self.p1.signature();
        let mut p2_sigs = self.p2.signature();
        p2_sigs.append(p1_sigs.as_mut());
        p2_sigs
    }
}

pub struct Points<F: BfBaseField> {
    p1: Point<F>,
    p2: Point<F>,
}

impl<F: BfBaseField> Points<F> {
    pub fn new(x1: F, y1: F, x2: F, y2: F) -> Points<F> {
        let p1 = Point::<F>::new(x1, y1);
        let p2 = Point::<F>::new(x2, y2);
        Self { p1, p2 }
    }

    pub fn recover_points_euqal_to_commited_points(&self) -> Script {
        let scripts = script! {
            {self.p1.recover_point_euqal_to_commited_point()}
            {self.p2.recover_point_euqal_to_commited_point()}
        };
        scripts
    }

    pub fn signature(&self) -> Vec<Vec<u8>> {
        let mut p1_sigs = self.p1.signature();
        let mut p2_sigs = self.p2.signature();
        p2_sigs.append(p1_sigs.as_mut());
        p2_sigs
    }
}

pub struct ExtensionPoint<F: BfBaseField, EF: BfExtensionField<F>> {
    x: EF,
    y: EF,
    x_commit: BitCommitExtension<F, EF>,
    y_commit: BitCommitExtension<F, EF>,
}

impl<F: BfBaseField, EF: BfExtensionField<F>> ExtensionPoint<F, EF> {
    pub fn new_from_assign(
        x: EF,
        y: EF,
        bc_assign: &mut ExtensionBCAssignment<F, EF>,
    ) -> ExtensionPoint<F, EF> {
        let x_commit = bc_assign.assign(x.clone());
        let x_commit = x_commit.clone();
        let y_commit = bc_assign.assign(y.clone());
        let y_commit = y_commit.clone();
        // let y_commit = bc_assign.assign_extension1::<EF>(y);
        Self {
            x: x,
            y: y,
            x_commit: x_commit,
            y_commit: y_commit,
        }
    }

    pub fn new(x: EF, y: EF) -> ExtensionPoint<F, EF> {
        let x_commit =
            BitCommitExtension::<F, EF>::new("b138982ce17ac813d505b5b40b665d404e9528e8", x.clone());
        let y_commit =
            BitCommitExtension::<F, EF>::new("b138982ce17ac813d505b5b40b665d404e9528e8", y.clone());
        Self {
            x: x,
            y: y,
            x_commit: x_commit,
            y_commit: y_commit,
        }
    }

    pub fn recover_point_euqal_to_commited_point(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_euqal_to_commit_message() }
            { self.y_commit.recover_message_euqal_to_commit_message() }
        };

        scripts
    }

    pub fn recover_point_x_at_altstack_y_at_stack(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_at_altstack() }
            { self.y_commit.recover_message_at_stack() }
        };

        scripts
    }

    pub fn recover_point_at_altstack(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_at_altstack() }
            { self.y_commit.recover_message_at_altstack() }
        };

        scripts
    }

    pub fn recover_point_at_stack(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_at_stack() }
            { self.y_commit.recover_message_at_stack() }
        };

        scripts
    }

    pub fn signature(&self) -> Vec<Vec<u8>> {
        let mut x_sigs = self.x_commit.signature();
        let mut y_sigs = self.y_commit.signature();
        y_sigs.append(x_sigs.as_mut());
        y_sigs
    }
}

pub struct Point<F: BfBaseField> {
    x: F,
    y: F,
    x_commit: BitCommit<F>,
    y_commit: BitCommit<F>,
}

impl<F: BfBaseField> Point<F> {
    pub fn new_from_assign(x: F, y: F, bc_assign: &mut BCAssignment<F>) -> Point<F> {
        let commits = bc_assign.assign_multi(vec![x, y]);
        Self {
            x: x,
            y: y,
            x_commit: commits[0].clone(),
            y_commit: commits[1].clone(),
        }
    }

    pub fn new(x: F, y: F) -> Point<F> {
        let x_commit = BitCommit::<F>::new("b138982ce17ac813d505b5b40b665d404e9528e8", x);
        let y_commit = BitCommit::<F>::new("b138982ce17ac813d505b5b40b665d404e9528e8", y);
        Self {
            x: x,
            y: y,
            x_commit: x_commit,
            y_commit: y_commit,
        }
    }

    pub fn recover_point_euqal_to_commited_point(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_euqal_to_commit_message() }
            { self.y_commit.recover_message_euqal_to_commit_message() }
        };

        scripts
    }

    pub fn recover_point_x_at_altstack_y_at_stack(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_at_altstack() }
            { self.y_commit.recover_message_at_stack() }
        };

        scripts
    }

    pub fn recover_point_at_altstack(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_at_altstack() }
            { self.y_commit.recover_message_at_altstack() }
        };

        scripts
    }

    pub fn recover_point_at_stack(&self) -> Script {
        let scripts = script! {
            { self.x_commit.recover_message_at_stack() }
            { self.y_commit.recover_message_at_stack() }
        };

        scripts
    }

    pub fn signature(&self) -> Vec<Vec<u8>> {
        let mut x_sigs = self.x_commit.signature();
        let mut y_sigs = self.y_commit.signature();
        y_sigs.append(x_sigs.as_mut());
        y_sigs
    }
}

#[cfg(test)]
mod test {
    use p3_baby_bear::BabyBear;
    use p3_field::{AbstractExtensionField, AbstractField, PrimeField32};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    use super::*;
    use crate::fri::field::BfField;
    use crate::{execute_script_with_inputs, BaseCanCommit, BitCommitExtension};

    type F = BabyBear;
    type EF = p3_field::extension::BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn test_point_babybear() {
        use crate::BabyBear;
        let p = Point::<BabyBear>::new(BabyBear::from_u32(1), BabyBear::from_u32(2));

        let script = script! {
            {p.recover_point_euqal_to_commited_point()}
            OP_1
        };
        let inputs = p.signature();
        let res = execute_script_with_inputs(script, inputs);
        assert!(res.success);
    }

    #[test]
    fn test_point_Babybear4() {
        use super::*;
        let mut rng = ChaCha20Rng::seed_from_u64(0u64);
        let a = rng.gen::<EF>();
        let b = rng.gen::<EF>();

        let p = ExtensionPoint::<F, EF>::new(a, b);

        let script = script! {
            {p.recover_point_euqal_to_commited_point()}
            OP_1
        };
        let inputs = p.signature();
        let res = execute_script_with_inputs(script, inputs);
        assert!(res.success);
    }

    #[test]
    fn test_points_Babybear() {
        use crate::BabyBear;
        let p = Points::<BabyBear>::new(
            BabyBear::from_u32(1),
            BabyBear::from_u32(2),
            BabyBear::from_u32(3),
            BabyBear::from_u32(4),
        );

        let script = script! {
            {p.recover_points_euqal_to_commited_points()}
            OP_1
        };
        let inputs = p.signature();
        let res = execute_script_with_inputs(script, inputs);
        assert!(res.success);
    }

    #[test]
    fn test_extension_points_Babybear4() {
        use super::*;
        let mut rng = ChaCha20Rng::seed_from_u64(0u64);
        let a = rng.gen::<EF>();
        let b = rng.gen::<EF>();
        let c = rng.gen::<EF>();
        let d = rng.gen::<EF>();

        let p = ExtensionPoints::<F, EF>::new(a, b, c, d);

        let script = script! {
            {p.recover_points_euqal_to_commited_points()}
            OP_1
        };
        let inputs = p.signature();
        let res = execute_script_with_inputs(script, inputs);
        assert!(res.success);
    }
}