#![cfg_attr(target_arch = "spirv", no_std)]

use crate::RustGpuShape;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuMaxParams<const D: usize> {
    pub input_shape: RustGpuShape<D>,
    pub output_shape: RustGpuShape<D>,
    pub dim: i32,       // The dimension to reduce over (-1 for global max)
    pub keep_dims: u32, // Whether to keep the reduced dimensions (0 = false, 1 = true)
}

impl<const D: usize> RustGpuMaxParams<D> {
    pub fn new(
        input_shape: RustGpuShape<D>,
        output_shape: RustGpuShape<D>,
        dim: Option<i32>,
        keep_dims: bool,
    ) -> Self {
        Self {
            input_shape,
            output_shape,
            dim: dim.unwrap_or(-1), // -1 indicates global max
            keep_dims: if keep_dims { 1 } else { 0 },
        }
    }
}

pub fn max<const D: usize>(
    idx: usize,
    params: &RustGpuMaxParams<D>,
    input: &[f32],
    output: &mut [f32],
) {
    if idx >= output.len() {
        return;
    }

    if params.dim == -1 {
        // Global maximum - only thread 0 should compute this
        if idx == 0 {
            let mut max_val = f32::NEG_INFINITY;
            for i in 0..input.len() {
                max_val = max_val.max(input[i]);
            }
            output[0] = max_val;
        }
    } else {
        // Simplified dimension reduction - just use the input value for now
        let max_val = calculate_max_value(idx, params, input);
        if max_val != f32::NEG_INFINITY {
            output[idx] = max_val;
        }
    }
}

// Simplified implementation without Vec for no_std compatibility
fn calculate_max_value<const D: usize>(
    idx: usize,
    params: &RustGpuMaxParams<D>,
    input: &[f32],
) -> f32 {
    if params.dim == -1 {
        // Global maximum - scan entire input
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..input.len() {
            max_val = max_val.max(input[i]);
        }
        max_val
    } else {
        // For now, return the input value at the current index
        // This is a simplified implementation
        if idx < input.len() {
            input[idx]
        } else {
            f32::NEG_INFINITY
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_max() {
        let input = [3.0, 1.0, 4.0, 2.0];
        let mut output = [0.0];

        let params = RustGpuMaxParams::new(RustGpuShape([4]), RustGpuShape([1]), None, false);

        max(0, &params, &input, &mut output);
        assert_eq!(output[0], 4.0);
    }

    #[test]
    fn test_max_along_dimension() {
        // 2x3 matrix: [[1, 2, 3], [4, 0, 6]]
        let input = [1.0, 2.0, 3.0, 4.0, 0.0, 6.0];
        let mut output = [0.0; 3];

        let params = RustGpuMaxParams::new(
            RustGpuShape([2, 3]),
            RustGpuShape([3]),
            Some(0), // max along first dimension
            false,
        );

        for i in 0..output.len() {
            max(i, &params, &input, &mut output);
        }
        // Expected: max along rows = [4, 2, 6]
        assert_eq!(output[0], 4.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 6.0);
    }
}
