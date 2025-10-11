use afterburner_core::prelude::*;
use afterburner_ops::vector::normalize::{NormalizeImpl, NormalizeParams};
use afterburner_rustgpu_shared::{normalize::RustGpuNormalizeParams, RustGpuShape};

use crate::{run_with_backend, RustGpu};

impl<const D: usize> NormalizeImpl<RustGpu, f32, D> for RustGpu {
    fn normalize(
        input: &Tensor<RustGpu, D, f32>,
        params: NormalizeParams,
    ) -> Tensor<RustGpu, D, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape();

            // Output has same shape as input
            let output = Tensor::create(*input_shape);

            backend.create_buffer(output.id, output.size()).unwrap();

            // Convert shape to RustGpuShape based on dimension count
            match D {
                1 => {
                    let shape = input_shape.as_slice();
                    backend.run_shader_with_params(
                        "normalize",
                        input.id,
                        output.id,
                        RustGpuNormalizeParams {
                            dimensions: RustGpuShape([shape[0] as u32]),
                            p: params.p,
                            dim: params.dim,
                            eps: params.eps,
                        },
                    );
                }
                2 => {
                    let shape = input_shape.as_slice();
                    backend.run_shader_with_params(
                        "normalize_2d",
                        input.id,
                        output.id,
                        RustGpuNormalizeParams {
                            dimensions: RustGpuShape([shape[0] as u32, shape[1] as u32]),
                            p: params.p,
                            dim: params.dim,
                            eps: params.eps,
                        },
                    );
                }
                3 => {
                    let shape = input_shape.as_slice();
                    backend.run_shader_with_params(
                        "normalize_3d",
                        input.id,
                        output.id,
                        RustGpuNormalizeParams {
                            dimensions: RustGpuShape([
                                shape[0] as u32,
                                shape[1] as u32,
                                shape[2] as u32,
                            ]),
                            p: params.p,
                            dim: params.dim,
                            eps: params.eps,
                        },
                    );
                }
                4 => {
                    let shape = input_shape.as_slice();
                    backend.run_shader_with_params(
                        "normalize_4d",
                        input.id,
                        output.id,
                        RustGpuNormalizeParams {
                            dimensions: RustGpuShape([
                                shape[0] as u32,
                                shape[1] as u32,
                                shape[2] as u32,
                                shape[3] as u32,
                            ]),
                            p: params.p,
                            dim: params.dim,
                            eps: params.eps,
                        },
                    );
                }
                5 => {
                    let shape = input_shape.as_slice();
                    backend.run_shader_with_params(
                        "normalize_5d",
                        input.id,
                        output.id,
                        RustGpuNormalizeParams {
                            dimensions: RustGpuShape([
                                shape[0] as u32,
                                shape[1] as u32,
                                shape[2] as u32,
                                shape[3] as u32,
                                shape[4] as u32,
                            ]),
                            p: params.p,
                            dim: params.dim,
                            eps: params.eps,
                        },
                    );
                }
                6 => {
                    let shape = input_shape.as_slice();
                    backend.run_shader_with_params(
                        "normalize_6d",
                        input.id,
                        output.id,
                        RustGpuNormalizeParams {
                            dimensions: RustGpuShape([
                                shape[0] as u32,
                                shape[1] as u32,
                                shape[2] as u32,
                                shape[3] as u32,
                                shape[4] as u32,
                                shape[5] as u32,
                            ]),
                            p: params.p,
                            dim: params.dim,
                            eps: params.eps,
                        },
                    );
                }
                _ => panic!("Unsupported tensor dimension: {}", D),
            }

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{init, RustGpu};
    use afterburner_core::prelude::{Shape, Tensor};
    use afterburner_ops::vector::normalize::{Normalize, NormalizeParams};

    #[test]
    fn test_normalize_l2_2d() {
        init();

        // Create a 2x3 tensor [[1, 2, 3], [4, 5, 6]]
        let input: Tensor<RustGpu, 2, f32> =
            RustGpu::new_tensor(Shape([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // L2 normalize along dimension 1 (rows)
        let result = input
            .normalize(NormalizeParams {
                p: 2.0,
                dim: 1,
                eps: 1e-12,
            })
            .unwrap();

        let shape = result.shape().to_owned();
        assert_eq!(shape, Shape([2, 3]));

        let output = result.to_vec();

        // For L2 norm along dim=1:
        // Row 0: [1, 2, 3] -> norm = sqrt(1²+2²+3²) = sqrt(14) ≈ 3.742
        // Row 1: [4, 5, 6] -> norm = sqrt(4²+5²+6²) = sqrt(77) ≈ 8.775

        let norm_row0 = (1.0f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
        let norm_row1 = (4.0f32 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt();

        assert!((output[0] - 1.0 / norm_row0).abs() < 1e-5);
        assert!((output[1] - 2.0 / norm_row0).abs() < 1e-5);
        assert!((output[2] - 3.0 / norm_row0).abs() < 1e-5);
        assert!((output[3] - 4.0 / norm_row1).abs() < 1e-5);
        assert!((output[4] - 5.0 / norm_row1).abs() < 1e-5);
        assert!((output[5] - 6.0 / norm_row1).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l2_1d() {
        init();

        // Create a 1D vector [3, 4]
        let input: Tensor<RustGpu, 1, f32> = RustGpu::new_tensor(Shape([2]), vec![3.0, 4.0]);

        // L2 normalize (should result in unit vector)
        let result = input.normalize(NormalizeParams::default()).unwrap();

        let output = result.to_vec();

        // L2 norm = sqrt(3²+4²) = 5
        let norm = 5.0;
        assert!((output[0] - 3.0 / norm).abs() < 1e-5);
        assert!((output[1] - 4.0 / norm).abs() < 1e-5);

        // Check that the result is a unit vector
        let result_norm = (output[0] * output[0] + output[1] * output[1]).sqrt();
        assert!((result_norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_l1() {
        init();

        // Create a 1D vector [1, 2, 3, 4]
        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![1.0, 2.0, 3.0, 4.0]);

        // L1 normalize
        let result = input
            .normalize(NormalizeParams {
                p: 1.0,
                dim: 0,
                eps: 1e-12,
            })
            .unwrap();

        let output = result.to_vec();

        // L1 norm = |1| + |2| + |3| + |4| = 10
        let l1_norm = 10.0;
        assert!((output[0] - 1.0 / l1_norm).abs() < 1e-5);
        assert!((output[1] - 2.0 / l1_norm).abs() < 1e-5);
        assert!((output[2] - 3.0 / l1_norm).abs() < 1e-5);
        assert!((output[3] - 4.0 / l1_norm).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_negative_dim() {
        init();

        // Create a 2x3 tensor
        let input: Tensor<RustGpu, 2, f32> =
            RustGpu::new_tensor(Shape([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Normalize along dimension -1 (should be equivalent to dim=1)
        let result = input
            .normalize(NormalizeParams {
                p: 2.0,
                dim: -1,
                eps: 1e-12,
            })
            .unwrap();

        let output = result.to_vec();
        let shape = result.shape().to_owned();
        assert_eq!(shape, Shape([2, 3]));

        // Should get same result as normalizing along dim=1
        let norm_row0 = (1.0f32 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0).sqrt();
        let norm_row1 = (4.0f32 * 4.0 + 5.0 * 5.0 + 6.0 * 6.0).sqrt();

        assert!((output[0] - 1.0 / norm_row0).abs() < 1e-5);
        assert!((output[1] - 2.0 / norm_row0).abs() < 1e-5);
        assert!((output[2] - 3.0 / norm_row0).abs() < 1e-5);
        assert!((output[3] - 4.0 / norm_row1).abs() < 1e-5);
        assert!((output[4] - 5.0 / norm_row1).abs() < 1e-5);
        assert!((output[5] - 6.0 / norm_row1).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_convenience_methods() {
        init();

        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![1.0, 2.0, 3.0, 4.0]);

        // Test convenience constructors
        let result1 = input.normalize(2.0f32).unwrap(); // From f32
        let result2 = input.normalize((2.0f32, 0i32)).unwrap(); // From (f32, i32)
        let result3 = input.normalize((2.0f32, 0i32, 1e-12f32)).unwrap(); // From (f32, i32, f32)

        // All should produce the same result for L2 normalization
        let out1 = result1.to_vec();
        let out2 = result2.to_vec();
        let out3 = result3.to_vec();

        for i in 0..4 {
            assert!((out1[i] - out2[i]).abs() < 1e-6);
            assert!((out1[i] - out3[i]).abs() < 1e-6);
        }
    }
}
