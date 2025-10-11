use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct LeakyReLUParams {
    pub alpha: f32,
}

impl Default for LeakyReLUParams {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

pub trait LeakyReLUImpl<B: Backend, T: Clone> {
    fn leaky_relu(input: &Tensor<B, 4, T>, params: LeakyReLUParams) -> Tensor<B, 4, T>;
}

pub trait LeakyReLU<B: Backend, T: Clone> {
    fn leaky_relu(&self, alpha: f32) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + LeakyReLUImpl<B, T>, T: Clone> LeakyReLU<B, T> for Tensor<B, 4, T> {
    fn leaky_relu(&self, alpha: f32) -> AbResult<Tensor<B, 4, T>> {
        Ok(B::leaky_relu(self, LeakyReLUParams { alpha }))
    }
}

pub trait SigmoidImpl<B: Backend, T: Clone> {
    fn sigmoid(input: &Tensor<B, 4, T>) -> Tensor<B, 4, T>;
}

pub trait Sigmoid<B: Backend, T: Clone> {
    fn sigmoid(&self) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + SigmoidImpl<B, T>, T: Clone> Sigmoid<B, T> for Tensor<B, 4, T> {
    fn sigmoid(&self) -> AbResult<Tensor<B, 4, T>> {
        Ok(B::sigmoid(self))
    }
}
