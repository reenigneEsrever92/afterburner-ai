#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuLeakyReLUParams {
    pub alpha: f32,
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuSigmoidParams {
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuTanhParams {
    pub size: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RustGpuReLUParams {
    pub size: u32,
}

/// LeakyReLU activation function
/// f(x) = max(alpha * x, x)
pub fn leaky_relu(idx: usize, params: &RustGpuLeakyReLUParams, input: &[f32], output: &mut [f32]) {
    if idx >= params.size as usize {
        return;
    }

    let x = input[idx];
    output[idx] = if x > 0.0 { x } else { params.alpha * x };
}

/// Sigmoid activation function
/// f(x) = 1 / (1 + exp(-x))
pub fn sigmoid(idx: usize, params: &RustGpuSigmoidParams, input: &[f32], output: &mut [f32]) {
    if idx >= params.size as usize {
        return;
    }

    let x = input[idx];
    // Clamp input to prevent overflow
    let clamped_x = if x > 88.0 {
        88.0
    } else if x < -88.0 {
        -88.0
    } else {
        x
    };

    #[cfg(target_arch = "spirv")]
    {
        use spirv_std::num_traits::Float;
        output[idx] = 1.0 / (1.0 + Float::exp(-clamped_x));
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        output[idx] = 1.0 / (1.0 + (-clamped_x).exp());
    }
}

/// Tanh activation function
/// f(x) = tanh(x)
pub fn tanh(idx: usize, params: &RustGpuTanhParams, input: &[f32], output: &mut [f32]) {
    if idx >= params.size as usize {
        return;
    }

    let x = input[idx];
    // Clamp input to prevent overflow
    let clamped_x = if x > 88.0 {
        88.0
    } else if x < -88.0 {
        -88.0
    } else {
        x
    };

    #[cfg(target_arch = "spirv")]
    {
        use spirv_std::num_traits::Float;
        let exp_2x = Float::exp(2.0 * clamped_x);
        output[idx] = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
    #[cfg(not(target_arch = "spirv"))]
    {
        let exp_2x = (2.0 * clamped_x).exp();
        output[idx] = (exp_2x - 1.0) / (exp_2x + 1.0);
    }
}

/// ReLU activation function
/// f(x) = max(0, x)
pub fn relu(idx: usize, params: &RustGpuReLUParams, input: &[f32], output: &mut [f32]) {
    if idx >= params.size as usize {
        return;
    }

    let x = input[idx];
    output[idx] = if x > 0.0 { x } else { 0.0 };
}
