use crate::RustGpuShape;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuUpsampleParams {
    pub input_shape: RustGpuShape<4>, // [batch, channels, height, width]
    pub output_shape: RustGpuShape<4>, // [batch, channels, height*scale, width*scale]
    pub scale_factor: u32,
    pub mode: u32, // 0 = nearest, 1 = bilinear
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuConcatenateParams {
    pub input_a_shape: RustGpuShape<4>,
    pub input_b_shape: RustGpuShape<4>,
    pub output_shape: RustGpuShape<4>,
    pub axis: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuTransposeParams {
    pub input_shape: RustGpuShape<4>,
    pub output_shape: RustGpuShape<4>,
    pub dims: [u32; 4], // permutation of [0, 1, 2, 3]
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuSliceParams {
    pub input_shape: RustGpuShape<4>,
    pub output_shape: RustGpuShape<4>,
    pub start: [u32; 4],
    pub size: [u32; 4],
}

/// Nearest neighbor upsampling
pub fn upsample_nearest(
    idx: usize,
    params: &RustGpuUpsampleParams,
    input: &[f32],
    output: &mut [f32],
) {
    let output_shape = params.output_shape.as_slice();
    let input_shape = params.input_shape.as_slice();

    if idx >= (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) as usize {
        return;
    }

    // Calculate 4D coordinates from flat index
    let out_w = idx as u32 % output_shape[3];
    let out_h = (idx as u32 / output_shape[3]) % output_shape[2];
    let out_c = (idx as u32 / (output_shape[3] * output_shape[2])) % output_shape[1];
    let out_b = idx as u32 / (output_shape[3] * output_shape[2] * output_shape[1]);

    // Map to input coordinates (nearest neighbor)
    let in_h = out_h / params.scale_factor;
    let in_w = out_w / params.scale_factor;

    // Calculate input index
    let in_idx = (out_b * input_shape[1] * input_shape[2] * input_shape[3]
        + out_c * input_shape[2] * input_shape[3]
        + in_h * input_shape[3]
        + in_w) as usize;

    if in_idx < input.len() {
        output[idx] = input[in_idx];
    }
}

/// Bilinear upsampling
pub fn upsample_bilinear(
    idx: usize,
    params: &RustGpuUpsampleParams,
    input: &[f32],
    output: &mut [f32],
) {
    let output_shape = params.output_shape.as_slice();
    let input_shape = params.input_shape.as_slice();

    if idx >= (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) as usize {
        return;
    }

    // Calculate 4D coordinates from flat index
    let out_w = idx as u32 % output_shape[3];
    let out_h = (idx as u32 / output_shape[3]) % output_shape[2];
    let out_c = (idx as u32 / (output_shape[3] * output_shape[2])) % output_shape[1];
    let out_b = idx as u32 / (output_shape[3] * output_shape[2] * output_shape[1]);

    // Calculate input coordinates with sub-pixel precision
    let scale_h = input_shape[2] as f32 / output_shape[2] as f32;
    let scale_w = input_shape[3] as f32 / output_shape[3] as f32;

    let in_h_f = (out_h as f32 + 0.5) * scale_h - 0.5;
    let in_w_f = (out_w as f32 + 0.5) * scale_w - 0.5;

    #[cfg(target_arch = "spirv")]
    let (in_h0, in_w0) = {
        use spirv_std::num_traits::Float;
        (Float::floor(in_h_f) as i32, Float::floor(in_w_f) as i32)
    };
    #[cfg(not(target_arch = "spirv"))]
    let (in_h0, in_w0) = (in_h_f.floor() as i32, in_w_f.floor() as i32);
    let in_h1 = in_h0 + 1;
    let in_w1 = in_w0 + 1;

    let h_weight = in_h_f - in_h0 as f32;
    let w_weight = in_w_f - in_w0 as f32;

    // Clamp coordinates
    let in_h0 = in_h0.max(0).min(input_shape[2] as i32 - 1) as u32;
    let in_w0 = in_w0.max(0).min(input_shape[3] as i32 - 1) as u32;
    let in_h1 = in_h1.max(0).min(input_shape[2] as i32 - 1) as u32;
    let in_w1 = in_w1.max(0).min(input_shape[3] as i32 - 1) as u32;

    // Get four neighboring pixels
    let idx_00 = (out_b * input_shape[1] * input_shape[2] * input_shape[3]
        + out_c * input_shape[2] * input_shape[3]
        + in_h0 * input_shape[3]
        + in_w0) as usize;
    let idx_01 = (out_b * input_shape[1] * input_shape[2] * input_shape[3]
        + out_c * input_shape[2] * input_shape[3]
        + in_h0 * input_shape[3]
        + in_w1) as usize;
    let idx_10 = (out_b * input_shape[1] * input_shape[2] * input_shape[3]
        + out_c * input_shape[2] * input_shape[3]
        + in_h1 * input_shape[3]
        + in_w0) as usize;
    let idx_11 = (out_b * input_shape[1] * input_shape[2] * input_shape[3]
        + out_c * input_shape[2] * input_shape[3]
        + in_h1 * input_shape[3]
        + in_w1) as usize;

    if idx_00 < input.len() && idx_01 < input.len() && idx_10 < input.len() && idx_11 < input.len()
    {
        let val_00 = input[idx_00];
        let val_01 = input[idx_01];
        let val_10 = input[idx_10];
        let val_11 = input[idx_11];

        // Bilinear interpolation
        let val_0 = val_00 * (1.0 - w_weight) + val_01 * w_weight;
        let val_1 = val_10 * (1.0 - w_weight) + val_11 * w_weight;
        output[idx] = val_0 * (1.0 - h_weight) + val_1 * h_weight;
    }
}

/// Concatenate two tensors along specified axis
pub fn concatenate(
    idx: usize,
    params: &RustGpuConcatenateParams,
    input_a: &[f32],
    input_b: &[f32],
    output: &mut [f32],
) {
    let output_shape = params.output_shape.as_slice();
    let input_a_shape = params.input_a_shape.as_slice();
    let input_b_shape = params.input_b_shape.as_slice();

    if idx >= (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) as usize {
        return;
    }

    // Calculate 4D coordinates from flat index
    let out_w = idx as u32 % output_shape[3];
    let out_h = (idx as u32 / output_shape[3]) % output_shape[2];
    let out_c = (idx as u32 / (output_shape[3] * output_shape[2])) % output_shape[1];
    let out_b = idx as u32 / (output_shape[3] * output_shape[2] * output_shape[1]);

    let axis = params.axis;
    let coords = [out_b, out_c, out_h, out_w];

    // Determine which input tensor to read from based on concatenation axis
    let split_point = input_a_shape[axis as usize];

    if coords[axis as usize] < split_point {
        // Read from input_a
        let in_idx = (coords[0] * input_a_shape[1] * input_a_shape[2] * input_a_shape[3]
            + coords[1] * input_a_shape[2] * input_a_shape[3]
            + coords[2] * input_a_shape[3]
            + coords[3]) as usize;

        if in_idx < input_a.len() {
            output[idx] = input_a[in_idx];
        }
    } else {
        // Read from input_b, adjusting coordinate along concatenation axis
        let mut b_coords = coords;
        b_coords[axis as usize] -= split_point;

        let in_idx = (b_coords[0] * input_b_shape[1] * input_b_shape[2] * input_b_shape[3]
            + b_coords[1] * input_b_shape[2] * input_b_shape[3]
            + b_coords[2] * input_b_shape[3]
            + b_coords[3]) as usize;

        if in_idx < input_b.len() {
            output[idx] = input_b[in_idx];
        }
    }
}

/// Transpose tensor dimensions
pub fn transpose(idx: usize, params: &RustGpuTransposeParams, input: &[f32], output: &mut [f32]) {
    let output_shape = params.output_shape.as_slice();
    let input_shape = params.input_shape.as_slice();

    if idx >= (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) as usize {
        return;
    }

    // Calculate 4D coordinates from flat index
    let out_coords = [
        idx as u32 / (output_shape[1] * output_shape[2] * output_shape[3]),
        (idx as u32 / (output_shape[2] * output_shape[3])) % output_shape[1],
        (idx as u32 / output_shape[3]) % output_shape[2],
        idx as u32 % output_shape[3],
    ];

    // Map output coordinates to input coordinates using permutation
    let in_coords = [
        out_coords[params.dims[0] as usize],
        out_coords[params.dims[1] as usize],
        out_coords[params.dims[2] as usize],
        out_coords[params.dims[3] as usize],
    ];

    // Calculate input index
    let in_idx = (in_coords[0] * input_shape[1] * input_shape[2] * input_shape[3]
        + in_coords[1] * input_shape[2] * input_shape[3]
        + in_coords[2] * input_shape[3]
        + in_coords[3]) as usize;

    if in_idx < input.len() {
        output[idx] = input[in_idx];
    }
}

/// Slice a tensor
pub fn slice(idx: usize, params: &RustGpuSliceParams, input: &[f32], output: &mut [f32]) {
    let output_shape = params.output_shape.as_slice();
    let input_shape = params.input_shape.as_slice();

    if idx >= (output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]) as usize {
        return;
    }

    // Calculate 4D coordinates from flat index
    let out_w = idx as u32 % output_shape[3];
    let out_h = (idx as u32 / output_shape[3]) % output_shape[2];
    let out_c = (idx as u32 / (output_shape[3] * output_shape[2])) % output_shape[1];
    let out_b = idx as u32 / (output_shape[3] * output_shape[2] * output_shape[1]);

    // Map to input coordinates by adding start offsets
    let in_coords = [
        out_b + params.start[0],
        out_c + params.start[1],
        out_h + params.start[2],
        out_w + params.start[3],
    ];

    // Calculate input index
    let in_idx = (in_coords[0] * input_shape[1] * input_shape[2] * input_shape[3]
        + in_coords[1] * input_shape[2] * input_shape[3]
        + in_coords[2] * input_shape[3]
        + in_coords[3]) as usize;

    if in_idx < input.len() {
        output[idx] = input[in_idx];
    }
}
