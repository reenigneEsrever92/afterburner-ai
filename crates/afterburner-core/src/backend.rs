use crate::tensor::Tensor;

pub trait Backend: Clone {
    fn as_slice<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> &[T];
    fn new_from<B: Backend, const D: usize, T: Clone>(t: Tensor<B, D, T>, data: &[u8]);
}
