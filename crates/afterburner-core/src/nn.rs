use crate::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Conv2DParams {
    pub stride: Shape<2>,
}

pub trait Conv2DImpl<B: Backend, T: Clone> {
    fn conv_2d(
        tensor: &Tensor<B, 4, T>,
        weights: &Tensor<B, 4, T>,
        params: Conv2DParams,
    ) -> Tensor<B, 4, T>;
}

pub trait Conv2D<B: Backend, T: Clone> {
    fn conv_2d(
        &self,
        weights: &Tensor<B, 4, T>,
        params: impl Into<Conv2DParams>,
    ) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + Conv2DImpl<B, T>, T: Clone> Conv2D<B, T> for Tensor<B, 4, T> {
    fn conv_2d(
        &self,
        weights: &Tensor<B, 4, T>,
        params: impl Into<Conv2DParams>,
    ) -> AbResult<Tensor<B, 4, T>> {
        // second dimension must be the same (input channels) on both tensor and weights
        self.shape().match_channels(weights.shape())?;
        Ok(B::conv_2d(self, weights, params.into()))
    }
}
