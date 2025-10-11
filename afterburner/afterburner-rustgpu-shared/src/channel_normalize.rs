#![cfg_attr(target_arch = "spirv", no_std)]

use crate::RustGpuShape;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuChannelNormalizeParams {
    pub dimensions: RustGpuShape<4>, // [batch_size, channels, height, width]
    pub num_channels: u32,
    pub mean: [f32; 16], // Support up to 16 channels (extendable if needed)
    pub std: [f32; 16],  // Support up to 16 channels (extendable if needed)
}

impl RustGpuChannelNormalizeParams {
    pub fn new(dimensions: RustGpuShape<4>, mean: &[f32], std: &[f32]) -> Self {
        let mut mean_array = [0.0f32; 16];
        let mut std_array = [0.0f32; 16];

        let num_channels = mean.len().min(16);

        // Copy mean and std values into fixed-size arrays
        for i in 0..num_channels {
            mean_array[i] = mean[i];
            std_array[i] = std[i];
        }

        Self {
            dimensions,
            num_channels: num_channels as u32,
            mean: mean_array,
            std: std_array,
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
pub fn channel_normalize(
    input: &[f32],
    output: &mut [f32],
    params: &RustGpuChannelNormalizeParams,
) {
    let [batch_size, channels, height, width] = params.dimensions.0;
    let spatial_size = (height * width) as usize;
    let channel_size = spatial_size;
    let batch_stride = (channels as usize) * channel_size;

    for batch in 0..batch_size as usize {
        for channel in 0..channels as usize {
            let mean = params.mean[channel];
            let std_dev = params.std[channel];

            for spatial_idx in 0..spatial_size {
                let idx = batch * batch_stride + channel * channel_size + spatial_idx;
                let input_val = input[idx];
                let normalized = (input_val - mean) / std_dev;
                output[idx] = normalized;
            }
        }
    }
}

#[cfg(target_arch = "spirv")]
pub fn channel_normalize(
    idx: usize,
    params: &RustGpuChannelNormalizeParams,
    input: &[f32],
    output: &mut [f32],
) {
    if idx >= output.len() {
        return;
    }

    let [batch_size, channels, height, width] = params.dimensions.0;
    let spatial_size = (height * width) as usize;
    let channel_size = spatial_size;
    let batch_stride = (channels as usize) * channel_size;

    // Calculate which batch, channel, and spatial position this index belongs to
    let batch_idx = idx / batch_stride;
    let remainder = idx % batch_stride;
    let channel_idx = remainder / channel_size;

    if batch_idx >= batch_size as usize || channel_idx >= channels as usize {
        return;
    }

    let mean = params.mean[channel_idx];
    let std_dev = params.std[channel_idx];

    let input_val = input[idx];
    let normalized = (input_val - mean) / std_dev;
    output[idx] = normalized;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_channel_normalize() {
        let dimensions = RustGpuShape([1, 2, 2, 2]); // 1 batch, 2 channels, 2x2 spatial
        let mean = [0.5, 1.0];
        let std = [0.5, 2.0];

        let params = RustGpuChannelNormalizeParams::new(dimensions, &mean, &std);

        // Input: channel 0 has values [1, 2, 3, 4], channel 1 has values [5, 6, 7, 8]
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0; 8];

        channel_normalize(&input, &mut output, &params);

        // Channel 0: (input - 0.5) / 0.5
        // [1, 2, 3, 4] -> [(1-0.5)/0.5, (2-0.5)/0.5, (3-0.5)/0.5, (4-0.5)/0.5] = [1, 3, 5, 7]
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!((output[1] - 3.0).abs() < 1e-5);
        assert!((output[2] - 5.0).abs() < 1e-5);
        assert!((output[3] - 7.0).abs() < 1e-5);

        // Channel 1: (input - 1.0) / 2.0
        // [5, 6, 7, 8] -> [(5-1)/2, (6-1)/2, (7-1)/2, (8-1)/2] = [2, 2.5, 3, 3.5]
        assert!((output[4] - 2.0).abs() < 1e-5);
        assert!((output[5] - 2.5).abs() < 1e-5);
        assert!((output[6] - 3.0).abs() < 1e-5);
        assert!((output[7] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_params_construction() {
        let dimensions = RustGpuShape([2, 3, 4, 4]);
        let mean = [0.485, 0.456, 0.406]; // ImageNet means
        let std = [0.229, 0.224, 0.225]; // ImageNet stds

        let params = RustGpuChannelNormalizeParams::new(dimensions, &mean, &std);

        assert_eq!(params.num_channels, 3);
        assert_eq!(params.mean[0], 0.485);
        assert_eq!(params.mean[1], 0.456);
        assert_eq!(params.mean[2], 0.406);
        assert_eq!(params.std[0], 0.229);
        assert_eq!(params.std[1], 0.224);
        assert_eq!(params.std[2], 0.225);

        // Remaining values should be 0.0
        assert_eq!(params.mean[3], 0.0);
        assert_eq!(params.std[3], 0.0);
    }
}
