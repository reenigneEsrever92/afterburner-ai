use crate::{
    backend::{Backend, NoBackend},
    shape::Shape,
};

#[derive(Debug)]
pub struct Tensor<B: Backend, const D: usize, T: Clone> {
    pub(crate) backend: B,
    shape: Shape<D>,
    data: Vec<T>,
}

impl<B: Backend, const D: usize, T: Clone> Tensor<B, D, T> {
    pub fn empty(backend: B, shape: impl Into<Shape<D>>) -> Tensor<B, D, T> {
        let shape: Shape<D> = shape.into();
        let size = shape.0.iter().product();

        Tensor {
            backend,
            shape,
            data: Vec::with_capacity(size),
        }
    }

    pub fn copy_to<B2: Backend>(&self, backend: B2) -> Tensor<B2, D, T> {
        Tensor {
            backend,
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data.as_slice()
    }

    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

// TODO: create macro
impl<const D: usize> From<[f32; D]> for Tensor<NoBackend, 3, f32> {
    fn from(value: [f32; D]) -> Self {
        Tensor {
            backend: NoBackend,
            shape: Shape([1, 1, D]),
            data: Vec::from(value),
        }
    }
}

impl<const D: usize, const D2: usize> From<[[f32; D]; D2]> for Tensor<NoBackend, 4, f32> {
    fn from(value: [[f32; D]; D2]) -> Self {
        Tensor {
            backend: NoBackend,
            shape: Shape([1, 1, D2, D]),
            data: Vec::from(value).concat(),
        }
    }
}

impl<const D: usize, const D2: usize, const D3: usize, const D4: usize>
    From<[[[[f32; D]; D2]; D3]; D4]> for Tensor<NoBackend, 4, f32>
{
    fn from(value: [[[[f32; D]; D2]; D3]; D4]) -> Self {
        Tensor {
            backend: NoBackend,
            shape: Shape([D4, D3, D2, D]),
            data: Vec::from(value).concat().concat().concat(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    #[test]
    fn test_from_4d() {
        let tensor: Tensor<_, 4, _> = [
            [
                [
                    [100.0, 101.0, 102.0],
                    [103.0, 104.0, 105.0],
                    [106.0, 107.0, 108.0],
                ],
                [
                    [110.0, 111.0, 112.0],
                    [113.0, 114.0, 115.0],
                    [116.0, 117.0, 118.0],
                ],
            ],
            [
                [
                    [200.0, 201.0, 202.0],
                    [203.0, 204.0, 105.0],
                    [206.0, 207.0, 208.0],
                ],
                [
                    [210.0, 211.0, 212.0],
                    [213.0, 214.0, 215.0],
                    [216.0, 217.0, 218.0],
                ],
            ],
        ]
        .into();

        assert_eq!(
            tensor.as_slice(),
            &[
                100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 110.0, 111.0, 112.0,
                113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 200.0, 201.0, 202.0, 203.0, 204.0, 105.0,
                206.0, 207.0, 208.0, 210.0, 211.0, 212.0, 213.0, 214.0, 215.0, 216.0, 217.0, 218.0,
            ]
        );
    }
}
