#![cfg_attr(target_arch = "spirv", no_std)]

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RustGpuConv2DParams {
    pub dimensions: RustGpuShape<4>,
    pub conv: RustGpuShape<4>,
    pub stride: RustGpuShape<2>,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuShape<const D: usize>(pub [usize; D]);

impl<const D: usize> RustGpuShape<D> {
    pub fn as_slice(&self) -> &[usize; D] {
        &self.0
    }
}
