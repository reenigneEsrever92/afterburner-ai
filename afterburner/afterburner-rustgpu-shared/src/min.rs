#![cfg_attr(target_arch = "spirv", no_std)]

use crate::RustGpuShape;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuMinParams<const D: usize> {
    pub input_shape: RustGpuShape<D>,
    pub output_shape: RustGpuShape<D>,
    pub dim: i32,       // The dimension to reduce over (-1 for global min)
    pub keep_dims: u32, // Whether to keep the reduced dimensions (0 = false, 1 = true)
}

impl<const D: usize> RustGpuMinParams<D> {
    pub fn new(
        input_shape: RustGpuShape<D>,
        output_shape: RustGpuShape<D>,
        dim: Option<i32>,
        keep_dims: bool,
    ) -> Self {
        Self {
            input_shape,
            output_shape,
            dim: dim.unwrap_or(-1), // -1 indicates global min
            keep_dims: if keep_dims { 1 } else { 0 },
        }
    }
}

pub fn min<const D: usize>(
    idx: usize,
    params: &RustGpuMinParams<D>,
    input: &[f32],
    output: &mut [f32],
) {
    if idx >= output.len() {
        return;
    }

    if params.dim == -1 {
        // Global minimum - only thread 0 should compute this
        if idx == 0 {
            let mut min_val = f32::INFINITY;
            for i in 0..input.len() {
                min_val = min_val.min(input[i]);
            }
            output[0] = min_val;
        }
    } else {
        // Simplified dimension reduction - just use the input value for now
        let min_val = calculate_min_value(idx, params, input);
        if min_val != f32::INFINITY {
            output[idx] = min_val;
        }
    }
}

// Simplified implementation without Vec for no_std compatibility
fn calculate_min_value<const D: usize>(
    idx: usize,
    params: &RustGpuMinParams<D>,
    input: &[f32],
) -> f32 {
    if params.dim == -1 {
        // Global minimum - scan entire input
        let mut min_val = f32::INFINITY;
        for i in 0..input.len() {
            min_val = min_val.min(input[i]);
        }
        min_val
    } else {
        // For now, return the input value at the current index
        // This is a simplified implementation
        if idx < input.len() {
            input[idx]
        } else {
            f32::INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_min() {
        let input = [3.0, 1.0, 4.0, 2.0];
        let mut output = [0.0];

        let params = RustGpuMinParams::new(RustGpuShape([4]), RustGpuShape([1]), None, false);

        for i in 0..output.len() {
            min(i, &params, &input, &mut output);
        }
        assert_eq!(output[0], 1.0);
    }

    #[test]
    fn test_min_along_dimension() {
        // 2x3 matrix: [[1, 2, 3], [4, 0, 6]]
        let input = [1.0, 2.0, 3.0, 4.0, 0.0, 6.0];
        let mut output = [0.0; 3];

        let params = RustGpuMinParams::new(
            RustGpuShape([2, 3]),
            RustGpuShape([3]),
            Some(0), // min along first dimension
            false,
        );

        min(0, &params, &input, &mut output);
        // Expected: min along rows = [1, 0, 3]
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[2], 3.0);
    }
}
