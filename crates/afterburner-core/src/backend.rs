use crate::{shape::Shape, tensor::Tensor};

pub trait Backend: Clone {
    fn read_tensor<const D: usize, T: Clone>(t: &Tensor<Self, D, T>) -> Vec<T>;
    fn new_tensor<const D: usize, T: Clone>(shape: Shape<D>, data: Vec<T>) -> Tensor<Self, D, T>;
}
