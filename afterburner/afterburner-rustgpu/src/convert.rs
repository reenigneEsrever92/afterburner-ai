use afterburner_core::prelude::*;

use crate::RustGpu;

impl<const D: usize> ConvertImpl<RustGpu, D, u8, f32> for RustGpu {
    fn convert(input: &Tensor<RustGpu, D, u8>) -> Tensor<RustGpu, D, f32> {
        todo!()
    }
}
