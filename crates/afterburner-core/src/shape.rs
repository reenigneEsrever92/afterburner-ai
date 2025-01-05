use crate::error::{AbResult, Error::ShapeMissmatch};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Shape<const D: usize>(pub(crate) [usize; D]);

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

    pub fn size(&self) -> usize {
        self.0.iter().product()
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
