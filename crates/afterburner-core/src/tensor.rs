use std::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering::Acquire},
};

use crate::{backend::Backend, cpu::Cpu, shape::Shape};

const counter: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
pub struct Tensor<B: Backend, const D: usize, T: Clone> {
    pub id: usize,
    pub backend: B,
    pub shape: Shape<D>,
    data: PhantomData<T>,
}

impl<B: Backend, const D: usize, T: Clone> Tensor<B, D, T> {
    pub fn new(backend: B, shape: impl Into<Shape<D>>) -> Self {
        Self {
            id: get_id(),
            backend,
            shape: shape.into(),
            data: PhantomData,
        }
    }

    pub fn copy_to<B2: Backend>(&self, backend: B2) -> Tensor<B2, D, T> {
        backend.new_tensor(self.shape, self.as_slice().to_vec())
    }

    pub fn as_slice(&self) -> &[T] {
        self.backend.as_slice(self)
    }

    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

fn get_id() -> usize {
    return counter.swap(*counter.get_mut() + 1, Acquire);
}

// TODO: create macro
impl<const D: usize> From<[f32; D]> for Tensor<Cpu, 3, f32> {
    fn from(value: [f32; D]) -> Self {
        Tensor::new(Cpu, Shape([1, 1, D]))
    }
}

impl<const D: usize, const D2: usize> From<[[f32; D]; D2]> for Tensor<Cpu, 4, f32> {
    fn from(value: [[f32; D]; D2]) -> Self {
        Tensor::new(Cpu, Shape([1, 1, D2, D]))
    }
}

impl<const D: usize, const D2: usize, const D3: usize, const D4: usize>
    From<[[[[f32; D]; D2]; D3]; D4]> for Tensor<Cpu, 4, f32>
{
    fn from(value: [[[[f32; D]; D2]; D3]; D4]) -> Self {
        Tensor::new(Cpu, Shape([D4, D3, D2, D]))
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
