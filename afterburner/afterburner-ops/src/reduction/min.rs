use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct MinParams {
    pub dim: Option<i32>, // The dimension to reduce over (None for global min)
    pub keep_dims: bool,  // Whether to keep the reduced dimensions
}

impl Default for MinParams {
    fn default() -> Self {
        Self {
            dim: None,
            keep_dims: false,
        }
    }
}

impl From<i32> for MinParams {
    fn from(dim: i32) -> Self {
        Self {
            dim: Some(dim),
            keep_dims: false,
        }
    }
}

impl From<(i32, bool)> for MinParams {
    fn from((dim, keep_dims): (i32, bool)) -> Self {
        Self {
            dim: Some(dim),
            keep_dims,
        }
    }
}

impl From<Option<i32>> for MinParams {
    fn from(dim: Option<i32>) -> Self {
        Self {
            dim,
            keep_dims: false,
        }
    }
}

pub trait MinImpl<B: Backend, T: Clone + PartialOrd, const D: usize> {
    fn min(input: &Tensor<B, D, T>, params: MinParams) -> Tensor<B, D, T>;
}

pub trait Min<B: Backend, T: Clone + PartialOrd, const D: usize> {
    fn min(&self, params: impl Into<MinParams>) -> AbResult<Tensor<B, D, T>>;

    // Convenience method for global minimum
    fn min_global(&self) -> AbResult<Tensor<B, D, T>> {
        self.min(MinParams {
            dim: None,
            keep_dims: false,
        })
    }
}

impl<B: Backend + MinImpl<B, T, D>, T: Clone + PartialOrd, const D: usize> Min<B, T, D>
    for Tensor<B, D, T>
{
    fn min(&self, params: impl Into<MinParams>) -> AbResult<Tensor<B, D, T>> {
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

        Ok(B::min(self, params))
    }
}
