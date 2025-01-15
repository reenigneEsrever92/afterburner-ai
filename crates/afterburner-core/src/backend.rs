use crate::tensor::Tensor;

pub trait Backend: Clone {
    fn as_slice<const D: usize, T: Clone>(&self, t: &Tensor<Self, D, T>) -> &[T];
    fn new_form<B: Backend, const D: usize, T: Clone>(
        &self,
        t: Tensor<B, D, T>,
    ) -> Tensor<Self, D, T>;
}
