use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct BatchNormParams {
    pub epsilon: f32,
}

impl Default for BatchNormParams {
    fn default() -> Self {
        Self { epsilon: 1e-5 }
    }
}

pub trait BatchNormImpl<B: Backend, T: Clone> {
    fn batch_norm(
        input: &Tensor<B, 4, T>,
        gamma: &Tensor<B, 1, T>,
        beta: &Tensor<B, 1, T>,
        params: BatchNormParams,
    ) -> Tensor<B, 4, T>;
}

pub trait BatchNorm<B: Backend, T: Clone> {
    fn batch_norm(
        &self,
        gamma: &Tensor<B, 1, T>,
        beta: &Tensor<B, 1, T>,
        params: impl Into<BatchNormParams>,
    ) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + BatchNormImpl<B, T>, T: Clone> BatchNorm<B, T> for Tensor<B, 4, T> {
    fn batch_norm(
        &self,
        gamma: &Tensor<B, 1, T>,
        beta: &Tensor<B, 1, T>,
        params: impl Into<BatchNormParams>,
    ) -> AbResult<Tensor<B, 4, T>> {
        // Gamma and beta should have same number of channels as input
        if gamma.shape().as_slice()[0] != self.shape().as_slice()[1] {
            return Err(Error::ShapeMissmatch);
        }
        if beta.shape().as_slice()[0] != self.shape().as_slice()[1] {
            return Err(Error::ShapeMissmatch);
        }
        Ok(B::batch_norm(self, gamma, beta, params.into()))
    }
}
