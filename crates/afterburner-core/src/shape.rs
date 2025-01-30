use crate::error::{AbResult, Error::ShapeMissmatch};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Shape<const D: usize>(pub [usize; D]);

impl<const D: usize> Default for Shape<D> {
    fn default() -> Self {
        Self([1; D])
    }
}

impl<const D: usize> Shape<D> {
    #[inline]
    pub fn as_slice(&self) -> &[usize; D] {
        &self.0
    }

    pub fn match_channels(&self, shape: &Shape<D>) -> AbResult<()> {
        if self.as_slice()[1] != shape.as_slice()[1] {
            Err(ShapeMissmatch)
        } else {
            Ok(())
        }
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.0.iter().product()
    }

    #[inline]
    pub fn size_for_dimension(&self, dim: usize) -> usize {
        let mut size = self.0[dim];
        for i in dim + 1..D {
            size *= self.0[i];
        }
        size
    }
}

impl<const D: usize> From<[usize; D]> for Shape<D> {
    fn from(value: [usize; D]) -> Self {
        Shape(value)
    }
}

impl<const D: usize> From<u64> for Shape<D> {
    fn from(value: u64) -> Self {
        Shape([value as usize; D])
    }
}
