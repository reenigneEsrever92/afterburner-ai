#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuElementwiseParams {
    pub size: u32,
}

/// Element-wise addition: output[i] = a[i] + b[i]
pub fn add(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    output[idx] = a[idx] + b[idx];
}

/// Element-wise subtraction: output[i] = a[i] - b[i]
pub fn sub(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    output[idx] = a[idx] - b[idx];
}

/// Element-wise multiplication: output[i] = a[i] * b[i]
pub fn mul(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    output[idx] = a[idx] * b[idx];
}

/// Element-wise division: output[i] = a[i] / b[i]
pub fn div(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    // Avoid division by zero
    #[cfg(target_arch = "spirv")]
    {
        use spirv_std::num_traits::Float;
        let divisor = if Float::abs(b[idx]) < 1e-8 {
            1e-8
        } else {
            b[idx]
        };
        output[idx] = a[idx] / divisor;
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let divisor = if b[idx].abs() < 1e-8 { 1e-8 } else { b[idx] };
        output[idx] = a[idx] / divisor;
    }
}

/// Element-wise power: output[i] = a[i] ^ b[i]
pub fn pow(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    #[cfg(target_arch = "spirv")]
    {
        use spirv_std::num_traits::Float;
        output[idx] = Float::powf(a[idx], b[idx]);
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        output[idx] = a[idx].powf(b[idx]);
    }
}

/// Element-wise maximum: output[i] = max(a[i], b[i])
pub fn max(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    #[cfg(target_arch = "spirv")]
    {
        use spirv_std::num_traits::Float;
        output[idx] = Float::max(a[idx], b[idx]);
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        output[idx] = a[idx].max(b[idx]);
    }
}

/// Element-wise minimum: output[i] = min(a[i], b[i])
pub fn min(
    idx: usize,
    params: &RustGpuElementwiseParams,
    a: &[f32],
    b: &[f32],
    output: &mut [f32],
) {
    if idx >= params.size as usize {
        return;
    }

    #[cfg(target_arch = "spirv")]
    {
        use spirv_std::num_traits::Float;
        output[idx] = Float::min(a[idx], b[idx]);
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        output[idx] = a[idx].min(b[idx]);
    }
}
