use afterburner_core::prelude::*;

#[derive(Debug, Clone)]
pub struct UpsampleParams {
    pub scale_factor: usize,
    pub mode: UpsampleMode,
}

#[derive(Debug, Clone)]
pub enum UpsampleMode {
    Nearest,
    Bilinear,
}

impl Default for UpsampleParams {
    fn default() -> Self {
        Self {
            scale_factor: 2,
            mode: UpsampleMode::Nearest,
        }
    }
}

pub trait UpsampleImpl<B: Backend, T: Clone> {
    fn upsample(input: &Tensor<B, 4, T>, params: UpsampleParams) -> Tensor<B, 4, T>;
}

pub trait Upsample<B: Backend, T: Clone> {
    fn upsample(&self, scale_factor: usize) -> AbResult<Tensor<B, 4, T>>;
    fn upsample_with_mode(
        &self,
        scale_factor: usize,
        mode: UpsampleMode,
    ) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + UpsampleImpl<B, T>, T: Clone> Upsample<B, T> for Tensor<B, 4, T> {
    fn upsample(&self, scale_factor: usize) -> AbResult<Tensor<B, 4, T>> {
        Ok(B::upsample(
            self,
            UpsampleParams {
                scale_factor,
                mode: UpsampleMode::Nearest,
            },
        ))
    }

    fn upsample_with_mode(
        &self,
        scale_factor: usize,
        mode: UpsampleMode,
    ) -> AbResult<Tensor<B, 4, T>> {
        Ok(B::upsample(self, UpsampleParams { scale_factor, mode }))
    }
}

#[derive(Debug, Clone)]
pub struct ConcatenateParams {
    pub axis: usize,
}

impl Default for ConcatenateParams {
    fn default() -> Self {
        Self { axis: 1 } // Default to channel dimension
    }
}

pub trait ConcatenateImpl<B: Backend, T: Clone> {
    fn concatenate(tensors: &[&Tensor<B, 4, T>], params: ConcatenateParams) -> Tensor<B, 4, T>;
}

pub trait Concatenate<B: Backend, T: Clone> {
    fn concatenate(&self, other: &Tensor<B, 4, T>, axis: usize) -> AbResult<Tensor<B, 4, T>>;
    fn concatenate_multi(tensors: &[&Tensor<B, 4, T>], axis: usize) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + ConcatenateImpl<B, T>, T: Clone> Concatenate<B, T> for Tensor<B, 4, T> {
    fn concatenate(&self, other: &Tensor<B, 4, T>, axis: usize) -> AbResult<Tensor<B, 4, T>> {
        if axis >= 4 {
            return Err(Error::ShapeMissmatch);
        }

        // Check that all dimensions except the concatenation axis match
        let self_shape = self.shape().as_slice();
        let other_shape = other.shape().as_slice();

        for i in 0..4 {
            if i != axis && self_shape[i] != other_shape[i] {
                return Err(Error::ShapeMissmatch);
            }
        }

        let tensors = vec![self, other];
        Ok(B::concatenate(&tensors, ConcatenateParams { axis }))
    }

    fn concatenate_multi(tensors: &[&Tensor<B, 4, T>], axis: usize) -> AbResult<Tensor<B, 4, T>> {
        if axis >= 4 {
            return Err(Error::ShapeMissmatch);
        }

        if tensors.is_empty() {
            return Err(Error::ShapeMissmatch);
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        // Check that all dimensions except the concatenation axis match
        let first_shape = tensors[0].shape().as_slice();
        for tensor in tensors.iter().skip(1) {
            let shape = tensor.shape().as_slice();
            for i in 0..4 {
                if i != axis && first_shape[i] != shape[i] {
                    return Err(Error::ShapeMissmatch);
                }
            }
        }

        Ok(B::concatenate(tensors, ConcatenateParams { axis }))
    }
}

pub trait TransposeImpl<B: Backend, T: Clone> {
    fn transpose(input: &Tensor<B, 4, T>, dims: [usize; 4]) -> Tensor<B, 4, T>;
}

pub trait Transpose<B: Backend, T: Clone> {
    fn transpose(&self, dims: [usize; 4]) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + TransposeImpl<B, T>, T: Clone> Transpose<B, T> for Tensor<B, 4, T> {
    fn transpose(&self, dims: [usize; 4]) -> AbResult<Tensor<B, 4, T>> {
        // Validate that dims is a permutation of [0, 1, 2, 3]
        let mut sorted_dims = dims;
        sorted_dims.sort();
        if sorted_dims != [0, 1, 2, 3] {
            return Err(Error::ShapeMissmatch);
        }

        Ok(B::transpose(self, dims))
    }
}

#[derive(Debug, Clone)]
pub struct SliceParams {
    pub start: [usize; 4],
    pub size: [usize; 4],
}

pub trait SliceImpl<B: Backend, T: Clone> {
    fn slice(input: &Tensor<B, 4, T>, params: SliceParams) -> Tensor<B, 4, T>;
}

pub trait Slice<B: Backend, T: Clone> {
    fn slice(&self, start: [usize; 4], size: [usize; 4]) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + SliceImpl<B, T>, T: Clone> Slice<B, T> for Tensor<B, 4, T> {
    fn slice(&self, start: [usize; 4], size: [usize; 4]) -> AbResult<Tensor<B, 4, T>> {
        let input_shape = self.shape().as_slice();

        // Validate slice bounds
        for i in 0..4 {
            if start[i] + size[i] > input_shape[i] {
                return Err(Error::ShapeMissmatch);
            }
        }

        Ok(B::slice(self, SliceParams { start, size }))
    }
}
