use crate::{backend::Backend, tensor::Tensor};

pub trait ConvertImpl<B: Backend, const D: usize, T: Clone, R: Clone> {
    fn convert(input: &Tensor<B, D, T>) -> Tensor<B, D, R>;
}

pub trait Convert<B: Backend, const D: usize, T: Clone, R: Clone> {
    fn convert(&self) -> Tensor<B, D, R>;
}

impl<B: Backend + ConvertImpl<B, D, T, R>, const D: usize, T: Clone, R: Clone> Convert<B, D, T, R>
    for Tensor<B, D, T>
{
    fn convert(&self) -> Tensor<B, D, R> {
        B::convert(self)
    }
}
