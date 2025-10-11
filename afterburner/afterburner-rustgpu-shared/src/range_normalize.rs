#![cfg_attr(target_arch = "spirv", no_std)]

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuRangeNormalizeParams {
    pub size: u32, // Total number of elements in the tensor
    pub eps: f32,  // Small epsilon to avoid division by zero
}

impl RustGpuRangeNormalizeParams {
    pub fn new(size: usize, eps: f32) -> Self {
        Self {
            size: size as u32,
            eps,
        }
    }
}

// High-performance range normalization: (value - min) / (max - min)
pub fn range_normalize(
    idx: usize,
    params: &RustGpuRangeNormalizeParams,
    input: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize || idx >= input.len() || idx >= output.len() {
        return;
    }

    // Two-pass algorithm:
    // Pass 1: Find global min and max (only thread 0 does this)
    // Pass 2: Normalize all values

    // For simplicity in GPU context, we'll do the computation per thread
    // In a real implementation, this would use shared memory and reduction
    if idx == 0 {
        // Thread 0 finds global min/max
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for i in 0..params.size as usize {
            if i < input.len() {
                min_val = min_val.min(input[i]);
                max_val = max_val.max(input[i]);
            }
        }

        let range = max_val - min_val;
        let safe_range = if range < params.eps {
            params.eps
        } else {
            range
        };

        // Normalize all values
        for i in 0..params.size as usize {
            if i < input.len() && i < output.len() {
                output[i] = (input[i] - min_val) / safe_range;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_normalize() {
        let input = [2.0, 4.0, 6.0, 8.0];
        let mut output = [0.0; 4];

        let params = RustGpuRangeNormalizeParams::new(4, 1e-8);

        range_normalize(0, &params, &input, &mut output);

        // Expected: (value - 2) / (8 - 2) = (value - 2) / 6
        // [2, 4, 6, 8] -> [0, 1/3, 2/3, 1]
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((output[2] - 2.0 / 3.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_range_normalize_same_values() {
        let input = [5.0, 5.0, 5.0, 5.0];
        let mut output = [0.0; 4];

        let params = RustGpuRangeNormalizeParams::new(4, 1e-6);

        range_normalize(0, &params, &input, &mut output);

        // When all values are the same, they should all become 0
        for &val in &output {
            assert!((val - 0.0).abs() < 1e-5);
        }
    }
}
