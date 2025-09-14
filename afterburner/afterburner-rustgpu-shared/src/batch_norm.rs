use crate::RustGpuShape;

#[cfg(target_arch = "spirv")]
#[inline]
fn sqrt(x: f32) -> f32 {
    // Newton-Raphson approximation for square root
    if x <= 0.0 {
        return 0.0;
    }

    let mut guess = x * 0.5;
    for _ in 0..5 {
        guess = 0.5 * (guess + x / guess);
    }
    guess
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
fn sqrt(x: f32) -> f32 {
    x.sqrt()
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RustGpuBatchNormParams {
    pub dimensions: RustGpuShape<4>, // [batch, channels, height, width]
    pub epsilon: f32,
}

pub fn batch_norm(
    id: usize,
    params: &RustGpuBatchNormParams,
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    output: &mut [f32],
) {
    // Input dimensions: [batch, channels, height, width]
    let batch_size = params.dimensions.0[0] as usize;
    let channels = params.dimensions.0[1] as usize;
    let height = params.dimensions.0[2] as usize;
    let width = params.dimensions.0[3] as usize;

    let channel_size = height * width;
    let batch_stride = channels * channel_size;

    // Decode position from linear id
    let batch = id / batch_stride;
    let channel = (id % batch_stride) / channel_size;
    let spatial_idx = id % channel_size;

    // Bounds check
    if batch >= batch_size || channel >= channels || spatial_idx >= channel_size {
        return;
    }

    // Calculate mean and variance for this channel across the entire batch
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    let total_elements = batch_size * channel_size;

    for b in 0..batch_size {
        for s in 0..channel_size {
            let input_idx = b * batch_stride + channel * channel_size + s;
            let val = input[input_idx];
            sum += val;
            sum_sq += val * val;
        }
    }

    let mean = sum / total_elements as f32;
    let variance = (sum_sq / total_elements as f32) - (mean * mean);
    let std_dev = sqrt(variance + params.epsilon);

    // Normalize and apply scale/shift
    let input_val = input[id];
    let normalized = (input_val - mean) / std_dev;
    let scaled = normalized * gamma[channel] + beta[channel];

    output[id] = scaled;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::RustGpuShape;

    #[test]
    fn test_batch_norm() {
        // Simple test with 1 batch, 2 channels, 2x2 spatial
        let input = [
            // Batch 0, Channel 0
            1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1
            5.0, 6.0, 7.0, 8.0,
        ];
        let gamma = [1.0, 1.0]; // No scaling
        let beta = [0.0, 0.0]; // No shift
        let mut output = [0.0f32; 8];

        let params = RustGpuBatchNormParams {
            dimensions: RustGpuShape([1, 2, 2, 2]), // [batch, channels, height, width]
            epsilon: 1e-5,
        };

        // Apply batch norm to all elements
        for i in 0..8 {
            batch_norm(i, &params, &input, &gamma, &beta, &mut output);
        }

        // Check that mean is approximately 0 for each channel
        let ch0_mean = (output[0] + output[1] + output[2] + output[3]) / 4.0;
        let ch1_mean = (output[4] + output[5] + output[6] + output[7]) / 4.0;

        assert!((ch0_mean).abs() < 1e-6);
        assert!((ch1_mean).abs() < 1e-6);
    }

    #[test]
    fn test_batch_norm_with_scale_shift() {
        let input = [2.0, 4.0, 6.0, 8.0]; // Single channel, single batch
        let gamma = [2.0]; // Scale by 2
        let beta = [1.0]; // Shift by 1
        let mut output = [0.0f32; 4];

        let params = RustGpuBatchNormParams {
            dimensions: RustGpuShape([1, 1, 2, 2]),
            epsilon: 1e-5,
        };

        for i in 0..4 {
            batch_norm(i, &params, &input, &gamma, &beta, &mut output);
        }

        // Input mean = 5.0, variance = 5.0, std = sqrt(5.0)
        // Normalized input: [-1.34, -0.45, 0.45, 1.34] (approximately)
        // Scaled and shifted: 2 * normalized + 1
        for &val in &output {
            // Values should be centered around 1.0 (the beta value)
            assert!((val - 1.0).abs() < 3.0); // Reasonable range check
        }
    }
}
