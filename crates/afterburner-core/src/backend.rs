use crate::{shape::Shape, tensor::Tensor};

pub trait Backend: Clone {
    fn as_slice<const D: usize, T: Clone>(&self, t: &Tensor<Self, D, T>) -> &[T];
    fn new_tensor<const D: usize, T: Clone>(
        &self,
        shape: Shape<D>,
        data: Vec<T>,
    ) -> Tensor<Self, D, T>;
}
