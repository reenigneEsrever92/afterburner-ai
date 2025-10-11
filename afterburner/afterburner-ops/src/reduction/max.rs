use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct MaxParams {
    pub dim: Option<i32>, // The dimension to reduce over (None for global max)
    pub keep_dims: bool,  // Whether to keep the reduced dimensions
}

impl Default for MaxParams {
    fn default() -> Self {
        Self {
            dim: None,
            keep_dims: false,
        }
    }
}

impl From<i32> for MaxParams {
    fn from(dim: i32) -> Self {
        Self {
            dim: Some(dim),
            keep_dims: false,
        }
    }
}

impl From<(i32, bool)> for MaxParams {
    fn from((dim, keep_dims): (i32, bool)) -> Self {
        Self {
            dim: Some(dim),
            keep_dims,
        }
    }
}

impl From<Option<i32>> for MaxParams {
    fn from(dim: Option<i32>) -> Self {
        Self {
            dim,
            keep_dims: false,
        }
    }
}

pub trait MaxImpl<B: Backend, T: Clone + PartialOrd, const D: usize> {
    fn max(input: &Tensor<B, D, T>, params: MaxParams) -> Tensor<B, D, T>;
}

pub trait Max<B: Backend, T: Clone + PartialOrd, const D: usize> {
    fn max(&self, params: impl Into<MaxParams>) -> AbResult<Tensor<B, D, T>>;

    // Convenience method for global maximum
    fn max_global(&self) -> AbResult<Tensor<B, D, T>> {
        self.max(MaxParams {
            dim: None,
            keep_dims: false,
        })
    }
}

impl<B: Backend + MaxImpl<B, T, D>, T: Clone + PartialOrd, const D: usize> Max<B, T, D>
    for Tensor<B, D, T>
{
    fn max(&self, params: impl Into<MaxParams>) -> AbResult<Tensor<B, D, T>> {
        let params = params.into();

        // Validate dimension parameter if specified
        if let Some(dim) = params.dim {
            let normalized_dim = if dim < 0 {
                (D as i32 + dim) as usize
            } else {
                dim as usize
            };

            if normalized_dim >= D {
                return Err(Error::InvalidDimension);
            }
        }

        Ok(B::max(self, params))
    }
}
