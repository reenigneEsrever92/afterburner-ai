use std::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

use crate::{backend::Backend, shape::Shape};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
pub struct Tensor<B: Backend, const D: usize, T: Clone> {
    pub id: usize,
    pub shape: Shape<D>,
    backend: PhantomData<B>,
    data: PhantomData<T>,
}

impl<B: Backend, const D: usize, T: Clone> Tensor<B, D, T> {
    pub fn create(shape: impl Into<Shape<D>>) -> Self {
        Self {
            id: COUNTER.fetch_add(1, SeqCst),
            shape: shape.into(),
            data: PhantomData,
            backend: PhantomData,
        }
    }

    pub fn copy_to<B2: Backend>(&self) -> Tensor<B2, D, T> {
        B2::new_tensor(self.shape, self.as_slice().to_vec())
    }

    pub fn as_slice(&self) -> &[T] {
        B::as_slice(self)
    }

    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }
}

// TODO: create macro
// impl<const D: usize> From<[f32; D]> for Tensor<Cpu, 3, f32> {
//     fn from(value: [f32; D]) -> Self {
//         Tensor::new(Cpu, Shape([1, 1, D]))
//     }
// }

// impl<const D: usize, const D2: usize> From<[[f32; D]; D2]> for Tensor<Cpu, 4, f32> {
//     fn from(value: [[f32; D]; D2]) -> Self {
//         Tensor::new(Cpu, Shape([1, 1, D2, D]))
//     }
// }

impl<B: Backend, const D: usize, const D2: usize, const D3: usize, const D4: usize>
    From<[[[[f32; D]; D2]; D3]; D4]> for Tensor<B, 4, f32>
{
    fn from(value: [[[[f32; D]; D2]; D3]; D4]) -> Self {
        let ptr = value.as_ptr() as *mut f32;
        let length = D * D2 * D3 * D4;
        let data = unsafe { Vec::from(std::slice::from_raw_parts(ptr, length)) };
        B::new_tensor(Shape([D4, D3, D2, D]), data)
    }
}
