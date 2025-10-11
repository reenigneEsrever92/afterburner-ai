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
    if params.dim == -1 {
        // Global maximum using parallel reduction
        global_max_reduction(idx, input, output);
    } else {
        // Dimension-specific reduction
        dimension_max_reduction(idx, params, input, output);
    }
}

// High-performance global max using parallel reduction
fn global_max_reduction(global_idx: usize, input: &[f32], output: &mut [f32]) {
    const WORKGROUP_SIZE: usize = 256;

    // Each thread processes multiple elements to maximize throughput
    let elements_per_thread = (input.len() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    let thread_id = global_idx % WORKGROUP_SIZE;
    let workgroup_id = global_idx / WORKGROUP_SIZE;

    if workgroup_id > 0 {
        return; // Only first workgroup participates in this simplified version
    }

    // Phase 1: Each thread finds maximum of its assigned elements
    let mut local_max = f32::NEG_INFINITY;
    let start_idx = thread_id * elements_per_thread;
    let end_idx = ((thread_id + 1) * elements_per_thread).min(input.len());

    for i in start_idx..end_idx {
        local_max = local_max.max(input[i]);
    }

    // Phase 2: Tree reduction within workgroup (simulated with sequential approach for no_std)
    // In a real GPU implementation, this would use shared memory and barriers
    if thread_id == 0 {
        let mut global_max = local_max;

        // Simulate collecting results from other threads in workgroup
        for tid in 1..WORKGROUP_SIZE {
            let other_start = tid * elements_per_thread;
            let other_end = ((tid + 1) * elements_per_thread).min(input.len());

            if other_start < input.len() {
                let mut other_max = f32::NEG_INFINITY;
                for i in other_start..other_end {
                    other_max = other_max.max(input[i]);
                }
                global_max = global_max.max(other_max);
            }
        }

        output[0] = global_max;
    }
}

// High-performance dimension-specific max reduction
fn dimension_max_reduction<const D: usize>(
    idx: usize,
    params: &RustGpuMaxParams<D>,
    input: &[f32],
    output: &mut [f32],
) {
    if idx >= output.len() {
        return;
    }

    // For dimension-specific reduction, we need to compute the max
    // along the specified dimension for this output position

    let dim = if params.dim < 0 {
        (D as i32 + params.dim) as usize
    } else {
        params.dim as usize
    };

    if dim >= D {
        return;
    }

    // Calculate strides for efficient indexing
    let mut input_strides = [1usize; D];
    let mut output_strides = [1usize; D];

    // Calculate input strides
    for i in (0..D - 1).rev() {
        input_strides[i] = input_strides[i + 1] * params.input_shape.0[i + 1] as usize;
    }

    // Calculate output strides
    for i in (0..D - 1).rev() {
        output_strides[i] = output_strides[i + 1] * params.output_shape.0[i + 1] as usize;
    }

    // Convert linear output index to multi-dimensional coordinates
    let mut output_coords = [0usize; D];
    let mut remaining_idx = idx;
    for i in 0..D {
        output_coords[i] = remaining_idx / output_strides[i];
        remaining_idx %= output_strides[i];
    }

    // Find maximum along the reduction dimension
    let mut max_val = f32::NEG_INFINITY;
    let reduce_size = params.input_shape.0[dim] as usize;

    for i in 0..reduce_size {
        // Create input coordinates from output coordinates
        let mut input_coords = output_coords;
        input_coords[dim] = i;

        // Convert multi-dimensional coordinates to linear index
        let mut input_idx = 0;
        for j in 0..D {
            input_idx += input_coords[j] * input_strides[j];
        }

        if input_idx < input.len() {
            max_val = max_val.max(input[input_idx]);
        }
    }

    if max_val != f32::NEG_INFINITY {
        output[idx] = max_val;
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
            RustGpuShape([1, 3]),
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
