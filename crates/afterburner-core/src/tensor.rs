use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering::Acquire},
        Mutex,
    },
};

use crate::{backend::Backend, shape::Shape};

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
        backend.new_form(self.clone())
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
