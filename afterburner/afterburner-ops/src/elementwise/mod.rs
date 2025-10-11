use afterburner_core::prelude::*;

pub trait AddImpl<B: Backend, T: Clone> {
    fn add(a: &Tensor<B, 4, T>, b: &Tensor<B, 4, T>) -> Tensor<B, 4, T>;
}

pub trait Add<B: Backend, T: Clone> {
    fn add(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + AddImpl<B, T>, T: Clone> Add<B, T> for Tensor<B, 4, T> {
    fn add(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>> {
        // Check that shapes are compatible for broadcasting or exact match
        if self.shape().as_slice() != other.shape().as_slice() {
            return Err(Error::ShapeMissmatch);
        }
        Ok(B::add(self, other))
    }
}

pub trait MulImpl<B: Backend, T: Clone> {
    fn mul(a: &Tensor<B, 4, T>, b: &Tensor<B, 4, T>) -> Tensor<B, 4, T>;
}

pub trait Mul<B: Backend, T: Clone> {
    fn mul(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + MulImpl<B, T>, T: Clone> Mul<B, T> for Tensor<B, 4, T> {
    fn mul(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>> {
        // Check that shapes are compatible for broadcasting or exact match
        if self.shape().as_slice() != other.shape().as_slice() {
            return Err(Error::ShapeMissmatch);
        }
        Ok(B::mul(self, other))
    }
}

pub trait SubImpl<B: Backend, T: Clone> {
    fn sub(a: &Tensor<B, 4, T>, b: &Tensor<B, 4, T>) -> Tensor<B, 4, T>;
}

pub trait Sub<B: Backend, T: Clone> {
    fn sub(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + SubImpl<B, T>, T: Clone> Sub<B, T> for Tensor<B, 4, T> {
    fn sub(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>> {
        // Check that shapes are compatible for broadcasting or exact match
        if self.shape().as_slice() != other.shape().as_slice() {
            return Err(Error::ShapeMissmatch);
        }
        Ok(B::sub(self, other))
    }
}

pub trait DivImpl<B: Backend, T: Clone> {
    fn div(a: &Tensor<B, 4, T>, b: &Tensor<B, 4, T>) -> Tensor<B, 4, T>;
}

pub trait Div<B: Backend, T: Clone> {
    fn div(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + DivImpl<B, T>, T: Clone> Div<B, T> for Tensor<B, 4, T> {
    fn div(&self, other: &Tensor<B, 4, T>) -> AbResult<Tensor<B, 4, T>> {
        // Check that shapes are compatible for broadcasting or exact match
        if self.shape().as_slice() != other.shape().as_slice() {
            return Err(Error::ShapeMissmatch);
        }
        Ok(B::div(self, other))
    }
}
