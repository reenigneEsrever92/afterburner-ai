use std::{
    fmt::Debug,
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

use crate::{
    backend::Backend,
    error::{AbResult, Error},
    shape::Shape,
};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
pub struct Tensor<B: Backend, const D: usize, T: Clone> {
    pub id: usize,
    pub shape: Shape<D>,
    backend: PhantomData<B>,
    data: PhantomData<T>,
}

impl<B: Backend, const D: usize, T: Clone> Debug for Tensor<B, D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.id)
            .field("shape", &self.shape)
            .finish()
    }
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

    /// Reshape the tensor to a new shape with potentially different dimensions.
    ///
    /// The total number of elements must remain the same. This method allows changing
    /// both the shape and the number of dimensions of a tensor while preserving all data.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The desired shape for the tensor. Can be provided as `Shape<D2>`,
    ///   `[usize; D2]`, or any type that implements `Into<Shape<D2>>`.
    ///
    /// # Returns
    ///
    /// Returns `Ok(Tensor<B, D2, T>)` if the reshape is successful, or `Err(Error::ShapeMissmatch)`
    /// if the total number of elements in the new shape doesn't match the current tensor.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use afterburner_core::prelude::*;
    ///
    /// // Reshape a 2x3 tensor to 1x6
    /// # fn example<B: Backend>() -> AbResult<()> {
    /// let tensor: Tensor<B, 2, f32> = Tensor::create([2, 3]);
    /// let reshaped = tensor.reshape([1, 6])?;
    /// assert_eq!(reshaped.shape().as_slice(), &[1, 6]);
    ///
    /// // Reshape a 4D tensor to 2D
    /// let tensor4d: Tensor<B, 4, f32> = Tensor::create([2, 3, 2, 2]);
    /// let tensor2d = tensor4d.reshape([6, 4])?; // 2*3*2*2 = 24 elements = 6*4
    /// assert_eq!(tensor2d.shape().as_slice(), &[6, 4]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// This method will return `Error::ShapeMissmatch` if the product of dimensions
    /// in the new shape doesn't equal the product of dimensions in the current shape.
    pub fn reshape<const D2: usize>(
        self,
        new_shape: impl Into<Shape<D2>>,
    ) -> AbResult<Tensor<B, D2, T>> {
        let new_shape = new_shape.into();

        // Validate that the total size remains the same
        if self.shape.size() != new_shape.size() {
            return Err(Error::ShapeMissmatch);
        }

        Ok(B::move_tensor(self, new_shape))
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

impl<B: Backend, const D: usize> From<[f32; D]> for Tensor<B, 1, f32> {
    fn from(value: [f32; D]) -> Self {
        let ptr = value.as_ptr() as *mut f32;
        let data = unsafe { Vec::from(std::slice::from_raw_parts(ptr, D)) };
        B::new_tensor(Shape([D]), data)
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
