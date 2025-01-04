use crate::{
    backend::{Backend, NoBackend},
    shape::Shape,
};

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

    pub fn copy_to<B2: Backend>(&self, backend: &B2) -> Tensor<B2, D, T> {
        Tensor {
            backend: backend.to_owned(),
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data.as_slice()
    }

    pub fn shape(&self) -> Shape<D> {
        self.shape.to_owned()
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

impl<const D: usize, const D2: usize> From<[[f32; D2]; D]> for Tensor<NoBackend, 4, f32> {
    fn from(value: [[f32; D2]; D]) -> Self {
        Tensor {
            backend: NoBackend,
            shape: Shape([1, 1, D2, D]),
            data: Vec::from(value).concat(),
        }
    }
}
