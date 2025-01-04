#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Shape<const D: usize>(pub(crate) [usize; D]);

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
