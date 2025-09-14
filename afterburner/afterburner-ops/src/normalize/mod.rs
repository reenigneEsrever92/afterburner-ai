use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct NormalizeParams {
    pub p: f32,   // The exponent value in the norm formulation (default: 2.0)
    pub dim: i32, // The dimension to normalize over (default: 1)
    pub eps: f32, // Small value to avoid division by zero (default: 1e-12)
}

impl Default for NormalizeParams {
    fn default() -> Self {
        Self {
            p: 2.0,
            dim: 1,
            eps: 1e-12,
        }
    }
}

pub trait NormalizeImpl<B: Backend, T: Clone, const D: usize> {
    fn normalize(input: &Tensor<B, D, T>, params: NormalizeParams) -> Tensor<B, D, T>;
}

pub trait Normalize<B: Backend, T: Clone, const D: usize> {
    fn normalize(&self, params: impl Into<NormalizeParams>) -> AbResult<Tensor<B, D, T>>;
}

impl<B: Backend + NormalizeImpl<B, T, D>, T: Clone, const D: usize> Normalize<B, T, D>
    for Tensor<B, D, T>
{
    fn normalize(&self, params: impl Into<NormalizeParams>) -> AbResult<Tensor<B, D, T>> {
        let params = params.into();

        // Validate dimension parameter
        let dim = if params.dim < 0 {
            (D as i32 + params.dim) as usize
        } else {
            params.dim as usize
        };

        if dim >= D {
            return Err(Error::InvalidDimension);
        }

        // Validate p parameter
        if params.p <= 0.0 {
            return Err(Error::InvalidParameter);
        }

        Ok(B::normalize(self, params))
    }
}
