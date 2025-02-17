#![no_std]

use afterburner_rustgpu_shared::*;
use spirv_std::{glam::UVec3, spirv};

pub mod conv2d;

#[spirv(compute(threads(64)))]
pub fn test(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = id.x as usize;
    output[idx] = 1.0;
}

#[spirv(compute(threads(64)))]
pub fn convert_u8_f32(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = id.x as usize;
    output[idx] = input[idx] as f32;
}

#[spirv(compute(threads(64)))]
pub fn conv2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuConv2DParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] weights: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    conv2d::conv2d(id.x as usize, &params, input, weights, output);
}
