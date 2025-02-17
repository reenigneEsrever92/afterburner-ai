use std::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

use crate::{backend::Backend, shape::Shape};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
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
        B2::new_tensor(self.shape, self.to_vec().to_vec())
    }

    pub fn to_vec(&self) -> Vec<T> {
        B::read_tensor(self)
    }

    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /**
     * Size in bytes.
     */
    #[inline]
    pub fn size(&self) -> usize {
        std::mem::size_of::<T>() * self.shape.size()
    }
}

impl<B: Backend, const D: usize, T: Clone> Drop for Tensor<B, D, T> {
    fn drop(&mut self) {
        B::delete_tensor(self);
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

impl<B: Backend> From<Vec<u8>> for Tensor<B, 1, u8> {
    fn from(value: Vec<u8>) -> Self {
        B::new_tensor(Shape([value.len()]), value)
    }
}

impl<B: Backend, const D: usize> From<[u8; D]> for Tensor<B, 1, u8> {
    fn from(value: [u8; D]) -> Self {
        let ptr = value.as_ptr() as *mut u8;
        let length = D;
        let data = unsafe { Vec::from(std::slice::from_raw_parts(ptr, length)) };
        B::new_tensor(Shape([D]), data)
    }
}

impl<B: Backend, const D: usize, const D2: usize> From<[[u8; D]; D2]> for Tensor<B, 2, u8> {
    fn from(value: [[u8; D]; D2]) -> Self {
        let ptr = value.as_ptr() as *mut u8;
        let length = D * D2;
        let data = unsafe { Vec::from(std::slice::from_raw_parts(ptr, length)) };
        B::new_tensor(Shape([D2, D]), data)
    }
}

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
