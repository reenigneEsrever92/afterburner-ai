use afterburner_core::prelude::*;

use crate::RustGpu;

impl<const D: usize> ConvertImpl<RustGpu, D, u8, f32> for RustGpu {
    fn convert(input: Tensor<B, D, T>) -> Tensor<B, D, R> {
        todo!()
    }
}
