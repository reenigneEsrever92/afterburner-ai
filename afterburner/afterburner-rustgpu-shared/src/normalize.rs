use crate::RustGpuShape;

#[cfg(target_arch = "spirv")]
#[inline]
fn pow(x: f32, p: f32) -> f32 {
    if p == 1.0 {
        x.abs()
    } else if p == 2.0 {
        x * x
    } else {
        // For other p values, use iterative approximation
        // This is a simplified implementation for GPU compatibility
        if x == 0.0 {
            return 0.0;
        }

        let abs_x = x.abs();
        if p == 3.0 {
            abs_x * abs_x * abs_x
        } else if p == 4.0 {
            let x2 = abs_x * abs_x;
            x2 * x2
        } else {
            // Simple iterative approximation for other values
            let mut result = 1.0;
            let mut base = abs_x;
            let mut exp = p as i32;

            while exp > 0 {
                if exp % 2 == 1 {
                    result *= base;
                }
                base *= base;
                exp /= 2;
            }
            result
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
fn pow(x: f32, p: f32) -> f32 {
    x.abs().powf(p)
}

#[cfg(target_arch = "spirv")]
#[inline]
fn powf(x: f32, p: f32) -> f32 {
    if p == 0.5 {
        sqrt(x)
    } else if p == 1.0 {
        x
    } else if p == 2.0 {
        x * x
    } else if p == 1.0 / 2.0 {
        sqrt(x)
    } else if p == 1.0 / 3.0 {
        // Cube root approximation using Newton's method
        if x <= 0.0 {
            return 0.0;
        }
        let mut guess = x / 3.0;
        for _ in 0..5 {
            let guess_cubed = guess * guess * guess;
            guess = (2.0 * guess + x / guess_cubed) / 3.0;
        }
        guess
    } else {
        // For other fractional powers, use simple approximation
        if x <= 0.0 {
            0.0
        } else if p > 0.0 && p < 1.0 {
            // Approximate fractional powers
            let int_part = p as i32;
            let frac_part = p - int_part as f32;

            let mut result = 1.0;
            let mut base = x;
            let mut exp = int_part;

            while exp > 0 {
                if exp % 2 == 1 {
                    result *= base;
                }
                base *= base;
                exp /= 2;
            }

            // Simple linear interpolation for fractional part
            if frac_part > 0.0 {
                result *= 1.0 + frac_part * (x - 1.0) / x;
            }

            result
        } else {
            x
        }
    }
}

#[cfg(not(target_arch = "spirv"))]
#[inline]
fn powf(x: f32, p: f32) -> f32 {
    x.powf(p)
}

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
pub struct RustGpuNormalizeParams<const D: usize> {
    pub dimensions: RustGpuShape<D>,
    pub p: f32,
    pub dim: i32,
    pub eps: f32,
}

pub fn normalize<const D: usize>(
    id: usize,
    params: &RustGpuNormalizeParams<D>,
    input: &[f32],
    output: &mut [f32],
) {
    let dims = &params.dimensions.0;
    let mut total_size: usize = 1;
    for i in 0..D {
        total_size *= dims[i] as usize;
    }

    if id >= total_size {
        return;
    }

    // Convert negative dimension to positive
    let dim = if params.dim < 0 {
        (D as i32 + params.dim) as usize
    } else {
        params.dim as usize
    };

    if dim >= D {
        return;
    }

    // Calculate strides for each dimension
    let mut strides = [0usize; D];
    if D > 0 {
        strides[D - 1] = 1;
        if D > 1 {
            for i in (0..D - 1).rev() {
                strides[i] = strides[i + 1] * dims[i + 1] as usize;
            }
        }
    }

    // Convert linear index to multi-dimensional indices
    let mut indices = [0usize; D];
    let mut remaining = id;
    for i in 0..D {
        indices[i] = remaining / strides[i];
        remaining %= strides[i];
    }

    // Calculate the base index for the vector we're normalizing
    let mut base_indices = indices;
    base_indices[dim] = 0;

    let mut base_id = 0;
    for i in 0..D {
        base_id += base_indices[i] * strides[i];
    }

    // Calculate Lp norm for the vector along the specified dimension
    let mut norm = 0.0f32;
    let dim_size = dims[dim] as usize;

    for i in 0..dim_size {
        let vector_id = base_id + i * strides[dim];
        let val = input[vector_id];
        norm += pow(val, params.p);
    }

    // Take the p-th root to get the Lp norm
    norm = powf(norm, 1.0 / params.p);

    // Avoid division by zero
    norm = norm.max(params.eps);

    // Normalize the current element
    let input_val = input[id];
    output[id] = input_val / norm;
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::RustGpuShape;

    #[test]
    fn test_normalize_l2_dim1() {
        // Test L2 normalization along dimension 1 (columns)
        // Input: 2x3 matrix [[1, 2, 3], [4, 5, 6]]
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0f32; 6];

        let params = RustGpuNormalizeParams {
            dimensions: RustGpuShape([2, 3]), // 2 rows, 3 columns
            p: 2.0,
            dim: 1, // Normalize along columns
            eps: 1e-12,
        };

        // Apply normalization to all elements
        for i in 0..6 {
            normalize(i, &params, &input, &mut output);
        }

        // For L2 norm along dim=1:
        // Row 0: [1, 2, 3] -> norm = sqrt(1²+2²+3²) = sqrt(14) ≈ 3.742
        // Row 1: [4, 5, 6] -> norm = sqrt(4²+5²+6²) = sqrt(77) ≈ 8.775

        let expected_norm_row0 = (1.0f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
        let expected_norm_row1 = (4.0f32 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt();

        // Check normalized values
        assert!((output[0] - 1.0 / expected_norm_row0).abs() < 1e-6);
        assert!((output[1] - 2.0 / expected_norm_row0).abs() < 1e-6);
        assert!((output[2] - 3.0 / expected_norm_row0).abs() < 1e-6);
        assert!((output[3] - 4.0 / expected_norm_row1).abs() < 1e-6);
        assert!((output[4] - 5.0 / expected_norm_row1).abs() < 1e-6);
        assert!((output[5] - 6.0 / expected_norm_row1).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_l1() {
        // Test L1 normalization
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];

        let params = RustGpuNormalizeParams {
            dimensions: RustGpuShape([1, 4]), // 1x4 vector
            p: 1.0,                           // L1 norm
            dim: 1,
            eps: 1e-12,
        };

        for i in 0..4 {
            normalize(i, &params, &input, &mut output);
        }

        // L1 norm = |1| + |2| + |3| + |4| = 10
        let l1_norm = 10.0;
        assert!((output[0] - 1.0 / l1_norm).abs() < 1e-6);
        assert!((output[1] - 2.0 / l1_norm).abs() < 1e-6);
        assert!((output[2] - 3.0 / l1_norm).abs() < 1e-6);
        assert!((output[3] - 4.0 / l1_norm).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_negative_dim() {
        // Test with negative dimension (should be converted to positive)
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut output = [0.0f32; 4];

        let params = RustGpuNormalizeParams {
            dimensions: RustGpuShape([2, 2]),
            p: 2.0,
            dim: -1, // Should be converted to dim=1
            eps: 1e-12,
        };

        for i in 0..4 {
            normalize(i, &params, &input, &mut output);
        }

        // Should normalize along the last dimension (columns)
        // Row 0: [1, 2] -> norm = sqrt(5)
        // Row 1: [3, 4] -> norm = sqrt(25) = 5
        let norm_row0 = (1.0f32 * 1.0 + 2.0 * 2.0).sqrt();
        let norm_row1 = (3.0f32 * 3.0 + 4.0 * 4.0).sqrt();

        assert!((output[0] - 1.0 / norm_row0).abs() < 1e-6);
        assert!((output[1] - 2.0 / norm_row0).abs() < 1e-6);
        assert!((output[2] - 3.0 / norm_row1).abs() < 1e-6);
        assert!((output[3] - 4.0 / norm_row1).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_with_eps() {
        // Test epsilon handling with very small values
        let input = [1e-15, 0.0, 1e-15, 0.0];
        let mut output = [0.0f32; 4];

        let params = RustGpuNormalizeParams {
            dimensions: RustGpuShape([2, 2]),
            p: 2.0,
            dim: 1,
            eps: 1e-12,
        };

        for i in 0..4 {
            normalize(i, &params, &input, &mut output);
        }

        // Since the norm will be very small, eps should prevent division by zero
        for &val in &output {
            assert!(val.is_finite());
            assert!(!val.is_nan());
        }
    }
}
