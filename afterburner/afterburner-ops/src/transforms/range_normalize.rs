use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct RangeNormalizeParams {
    pub eps: f32, // Small epsilon to avoid division by zero
}

impl Default for RangeNormalizeParams {
    fn default() -> Self {
        Self { eps: 1e-8 }
    }
}

impl From<f32> for RangeNormalizeParams {
    fn from(eps: f32) -> Self {
        Self { eps }
    }
}

pub trait RangeNormalizeImpl<B: Backend, T: Clone, const D: usize> {
    fn range_normalize(input: &Tensor<B, D, T>, params: RangeNormalizeParams) -> Tensor<B, D, T>;
}

pub trait RangeNormalize<B: Backend, T: Clone, const D: usize> {
    fn range_normalize(&self, params: impl Into<RangeNormalizeParams>)
        -> AbResult<Tensor<B, D, T>>;

    /// Convenience method with default epsilon
    fn range_normalize_default(&self) -> AbResult<Tensor<B, D, T>> {
        self.range_normalize(RangeNormalizeParams::default())
    }
}

impl<B: Backend + RangeNormalizeImpl<B, T, D>, T: Clone, const D: usize> RangeNormalize<B, T, D>
    for Tensor<B, D, T>
{
    fn range_normalize(
        &self,
        params: impl Into<RangeNormalizeParams>,
    ) -> AbResult<Tensor<B, D, T>> {
        let params = params.into();

        // Validate epsilon parameter
        if params.eps < 0.0 {
            return Err(Error::InvalidParameter);
        }

        Ok(B::range_normalize(self, params))
    }
}
