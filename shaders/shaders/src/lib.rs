#![no_std]
use afterburner_rustgpu_shared::*;

use spirv_std::glam::UVec3;
use spirv_std::spirv;

#[spirv(compute(threads(64)))]
pub fn conv2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuConv2DParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] weights: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = id.x as usize;
    let batch_size = params.dimensions.0[1] * params.dimensions.0[2] * params.dimensions.0[3];
    let batch = idx / batch_size;
    let channel_size = params.dimensions.0[2] * params.dimensions.0[3];
    let channel = (idx - batch_size) / channel_size;
    let output_y = (idx - batch_size - channel_size) / params.dimensions.0[3];
    let output_x = idx - batch_size - channel_size - params.dimensions.0[3];

    let output_idx = params.stride.0;
    output[0] = params.stride.0[0] as f32;
    output[1] = params.stride.0[1] as f32;
    // output[2] = constants.stride.0[1];
}
