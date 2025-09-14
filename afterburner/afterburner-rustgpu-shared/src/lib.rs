#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(generic_const_exprs)]

pub mod batch_norm;
pub mod conv2d;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuShape<const D: usize>(pub [u32; D]);

impl<const D: usize> RustGpuShape<D> {
    pub fn as_slice(&self) -> &[u32; D] {
        &self.0
    }
}
