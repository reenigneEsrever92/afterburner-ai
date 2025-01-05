use crate::{backend::Backend, error::AbResult, shape::Shape, tensor::Tensor};

pub trait Conv2DImpl<B: Backend, T: Clone> {
    fn conv_2d(
        &self,
        tensor: &Tensor<B, 4, T>,
        weights: &Tensor<B, 4, T>,
        stride: Shape<2>,
    ) -> Tensor<B, 4, T>;
}

pub trait Conv2D<B: Backend, T: Clone> {
    fn conv_2d(
        &self,
        weights: &Tensor<B, 4, T>,
        stride: impl Into<Shape<2>>,
    ) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + Conv2DImpl<B, T>, T: Clone> Conv2D<B, T> for Tensor<B, 4, T> {
    fn conv_2d(
        &self,
        weights: &Tensor<B, 4, T>,
        stride: impl Into<Shape<2>>,
    ) -> AbResult<Tensor<B, 4, T>> {
        // second dimension must be the same (input channels) on both tensor and weights
        self.shape().match_channels(weights.shape())?;
        Ok(self.backend.conv_2d(&self, &weights, stride.into()))
    }
}
