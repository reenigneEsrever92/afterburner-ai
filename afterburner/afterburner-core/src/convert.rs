use crate::{backend::Backend, tensor::Tensor};

pub trait ConvertImpl<B: Backend, const D: usize, T: Clone, R: Clone> {
    fn convert(input: Tensor<B, D, T>) -> Tensor<B, D, R>;
}
