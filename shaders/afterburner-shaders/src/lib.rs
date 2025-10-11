#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(generic_const_exprs)]

use afterburner_rustgpu_shared::activation::{
    RustGpuLeakyReLUParams, RustGpuReLUParams, RustGpuSigmoidParams, RustGpuTanhParams,
};
use afterburner_rustgpu_shared::batch_norm::RustGpuBatchNormParams;
use afterburner_rustgpu_shared::channel_normalize::RustGpuChannelNormalizeParams;
use afterburner_rustgpu_shared::conv2d::RustGpuConv2DParams;
use afterburner_rustgpu_shared::elementwise::RustGpuElementwiseParams;
use afterburner_rustgpu_shared::max::RustGpuMaxParams;
use afterburner_rustgpu_shared::min::RustGpuMinParams;
use afterburner_rustgpu_shared::normalize::RustGpuNormalizeParams;
use afterburner_rustgpu_shared::range_normalize::RustGpuRangeNormalizeParams;
use afterburner_rustgpu_shared::reshape::{
    RustGpuConcatenateParams, RustGpuSliceParams, RustGpuTransposeParams, RustGpuUpsampleParams,
};
use spirv_std::{glam::UVec3, spirv};

#[spirv(compute(threads(256)))]
pub fn test(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] _input: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        output[idx] = 1.0;
    }
}

#[spirv(compute(threads(64)))]
pub fn convert_u8_f32(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < input.len() && idx < output.len() {
        output[idx] = input[idx] as f32;
    }
}

#[spirv(compute(threads(64)))]
pub fn convert_f32_u8(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u8],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < input.len() && idx < output.len() {
        // Clamp to 0-255 range and convert
        let clamped = if input[idx] < 0.0 {
            0.0
        } else if input[idx] > 255.0 {
            255.0
        } else {
            input[idx]
        };
        output[idx] = clamped as u8;
    }
}

#[spirv(compute(threads(64)))]
pub fn convert_grayscale_to_rgb(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u8],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < input.len() {
        let rgb_idx = idx * 3;
        if rgb_idx + 2 < output.len() {
            // Clamp and convert to u8
            let clamped = if input[idx] < 0.0 {
                0.0
            } else if input[idx] > 1.0 {
                1.0
            } else {
                input[idx]
            };
            let value = (clamped * 255.0) as u8;
            output[rgb_idx] = value; // R
            output[rgb_idx + 1] = value; // G
            output[rgb_idx + 2] = value; // B
        }
    }
}

#[spirv(compute(threads(64)))]
pub fn convert_rgb_to_grayscale(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u8],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    let rgb_idx = idx * 3;
    if rgb_idx + 2 < input.len() && idx < output.len() {
        let r = input[rgb_idx] as f32;
        let g = input[rgb_idx + 1] as f32;
        let b = input[rgb_idx + 2] as f32;

        // Convert RGB to grayscale using luminance formula
        let gray = 0.299 * r + 0.587 * g + 0.114 * b;
        output[idx] = gray / 255.0; // Normalize to 0-1 range
    }
}

#[spirv(compute(threads(64)))]
pub fn conv2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuConv2DParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] weights: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::conv2d::conv2d(idx, &params, input, weights, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn batch_norm(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuBatchNormParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] gamma: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] beta: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::batch_norm::batch_norm(
            idx, &params, input, gamma, beta, output,
        );
    }
}

#[spirv(compute(threads(64)))]
pub fn normalize(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuNormalizeParams<1>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::normalize::normalize(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn normalize_2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuNormalizeParams<2>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::normalize::normalize(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn normalize_3d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuNormalizeParams<3>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::normalize::normalize(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn normalize_4d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuNormalizeParams<4>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::normalize::normalize(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn normalize_5d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuNormalizeParams<5>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::normalize::normalize(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn normalize_6d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuNormalizeParams<6>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::normalize::normalize(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn channel_normalize(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuChannelNormalizeParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::channel_normalize::channel_normalize(
            idx, params, input, output,
        );
    }
}

#[spirv(compute(threads(256)))]
pub fn min(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMinParams<1>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::min::min(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn min_2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMinParams<2>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::min::min(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn min_3d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMinParams<3>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::min::min(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn min_4d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMinParams<4>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::min::min(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn min_5d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMinParams<5>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::min::min(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn min_6d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMinParams<6>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::min::min(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn max(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMaxParams<1>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::max::max(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn max_2d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMaxParams<2>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::max::max(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn max_3d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMaxParams<3>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::max::max(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn max_4d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMaxParams<4>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::max::max(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn max_5d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMaxParams<5>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::max::max(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn max_6d(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuMaxParams<6>,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::max::max(idx, &params, input, output);
    }
}

#[spirv(compute(threads(256)))]
pub fn range_normalize(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuRangeNormalizeParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::range_normalize::range_normalize(idx, &params, input, output);
    }
}

// Activation functions
#[spirv(compute(threads(64)))]
pub fn leaky_relu(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuLeakyReLUParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::activation::leaky_relu(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn sigmoid(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuSigmoidParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::activation::sigmoid(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn tanh(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuTanhParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::activation::tanh(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn relu(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuReLUParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::activation::relu(idx, &params, input, output);
    }
}

// Element-wise operations
#[spirv(compute(threads(64)))]
pub fn elementwise_add(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuElementwiseParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input_a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::elementwise::add(idx, &params, input_a, input_b, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn elementwise_sub(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuElementwiseParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input_a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::elementwise::sub(idx, &params, input_a, input_b, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn elementwise_mul(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuElementwiseParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input_a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::elementwise::mul(idx, &params, input_a, input_b, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn elementwise_div(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuElementwiseParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input_a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::elementwise::div(idx, &params, input_a, input_b, output);
    }
}

// Reshape operations
#[spirv(compute(threads(64)))]
pub fn upsample_nearest(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuUpsampleParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::reshape::upsample_nearest(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn upsample_bilinear(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuUpsampleParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::reshape::upsample_bilinear(idx, &params, input, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn concatenate(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuConcatenateParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input_a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] input_b: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::reshape::concatenate(idx, &params, input_a, input_b, output);
    }
}

#[spirv(compute(threads(64)))]
pub fn transpose(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] params: &RustGpuTransposeParams,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
) {
    let idx = (id.y * 65535 + id.x) as usize;
    if idx < output.len() {
        afterburner_rustgpu_shared::reshape::transpose(idx, &params, input, output);
    }
}
