use afterburner_core::prelude::*;
use afterburner_ops::prelude::*;
use afterburner_rustgpu_shared::{min::RustGpuMinParams, RustGpuShape};

use crate::{run_with_backend, RustGpu};

impl<const D: usize> MinImpl<RustGpu, f32, D> for RustGpu {
    fn min(input: &Tensor<RustGpu, D, f32>, params: MinParams) -> Tensor<RustGpu, D, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape();

            // For simplicity, output has same shape as input for now
            let output_shape = *input_shape;

            let output = Tensor::create(output_shape);
            backend.create_buffer(output.id, output.size()).unwrap();

            // Convert shapes to RustGpuShape based on dimension count
            match D {
                1 => {
                    let input_shape_slice = input_shape.as_slice();
                    let output_shape_slice = output_shape.as_slice();
                    backend.run_shader_with_params(
                        "min",
                        input.id,
                        output.id,
                        RustGpuMinParams {
                            input_shape: RustGpuShape([input_shape_slice[0] as u32]),
                            output_shape: RustGpuShape([output_shape_slice[0] as u32]),
                            dim: params.dim.unwrap_or(-1),
                            keep_dims: if params.keep_dims { 1 } else { 0 },
                        },
                    );
                }
                2 => {
                    let input_shape_slice = input_shape.as_slice();
                    let output_shape_slice = output_shape.as_slice();
                    backend.run_shader_with_params(
                        "min_2d",
                        input.id,
                        output.id,
                        RustGpuMinParams {
                            input_shape: RustGpuShape([
                                input_shape_slice[0] as u32,
                                input_shape_slice[1] as u32,
                            ]),
                            output_shape: RustGpuShape([
                                output_shape_slice[0] as u32,
                                output_shape_slice[1] as u32,
                            ]),
                            dim: params.dim.unwrap_or(-1),
                            keep_dims: if params.keep_dims { 1 } else { 0 },
                        },
                    );
                }
                3 => {
                    let input_shape_slice = input_shape.as_slice();
                    let output_shape_slice = output_shape.as_slice();
                    backend.run_shader_with_params(
                        "min_3d",
                        input.id,
                        output.id,
                        RustGpuMinParams {
                            input_shape: RustGpuShape([
                                input_shape_slice[0] as u32,
                                input_shape_slice[1] as u32,
                                input_shape_slice[2] as u32,
                            ]),
                            output_shape: RustGpuShape([
                                output_shape_slice[0] as u32,
                                output_shape_slice[1] as u32,
                                output_shape_slice[2] as u32,
                            ]),
                            dim: params.dim.unwrap_or(-1),
                            keep_dims: if params.keep_dims { 1 } else { 0 },
                        },
                    );
                }
                4 => {
                    let input_shape_slice = input_shape.as_slice();
                    let output_shape_slice = output_shape.as_slice();
                    backend.run_shader_with_params(
                        "min_4d",
                        input.id,
                        output.id,
                        RustGpuMinParams {
                            input_shape: RustGpuShape([
                                input_shape_slice[0] as u32,
                                input_shape_slice[1] as u32,
                                input_shape_slice[2] as u32,
                                input_shape_slice[3] as u32,
                            ]),
                            output_shape: RustGpuShape([
                                output_shape_slice[0] as u32,
                                output_shape_slice[1] as u32,
                                output_shape_slice[2] as u32,
                                output_shape_slice[3] as u32,
                            ]),
                            dim: params.dim.unwrap_or(-1),
                            keep_dims: if params.keep_dims { 1 } else { 0 },
                        },
                    );
                }
                5 => {
                    let input_shape_slice = input_shape.as_slice();
                    let output_shape_slice = output_shape.as_slice();
                    backend.run_shader_with_params(
                        "min_5d",
                        input.id,
                        output.id,
                        RustGpuMinParams {
                            input_shape: RustGpuShape([
                                input_shape_slice[0] as u32,
                                input_shape_slice[1] as u32,
                                input_shape_slice[2] as u32,
                                input_shape_slice[3] as u32,
                                input_shape_slice[4] as u32,
                            ]),
                            output_shape: RustGpuShape([
                                output_shape_slice[0] as u32,
                                output_shape_slice[1] as u32,
                                output_shape_slice[2] as u32,
                                output_shape_slice[3] as u32,
                                output_shape_slice[4] as u32,
                            ]),
                            dim: params.dim.unwrap_or(-1),
                            keep_dims: if params.keep_dims { 1 } else { 0 },
                        },
                    );
                }
                6 => {
                    let input_shape_slice = input_shape.as_slice();
                    let output_shape_slice = output_shape.as_slice();
                    backend.run_shader_with_params(
                        "min_6d",
                        input.id,
                        output.id,
                        RustGpuMinParams {
                            input_shape: RustGpuShape([
                                input_shape_slice[0] as u32,
                                input_shape_slice[1] as u32,
                                input_shape_slice[2] as u32,
                                input_shape_slice[3] as u32,
                                input_shape_slice[4] as u32,
                                input_shape_slice[5] as u32,
                            ]),
                            output_shape: RustGpuShape([
                                output_shape_slice[0] as u32,
                                output_shape_slice[1] as u32,
                                output_shape_slice[2] as u32,
                                output_shape_slice[3] as u32,
                                output_shape_slice[4] as u32,
                                output_shape_slice[5] as u32,
                            ]),
                            dim: params.dim.unwrap_or(-1),
                            keep_dims: if params.keep_dims { 1 } else { 0 },
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
    use crate::prelude::*;

    #[test]
    fn test_min_global_1d() {
        init();

        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![3.0, 1.0, 4.0, 2.0]);

        let result = input.min_global().unwrap();
        let output = result.to_vec();

        assert_eq!(output.len(), 1);
        assert_eq!(output[0], 1.0);
    }

    #[test]
    fn test_min_along_dimension_2d() {
        init();

        // Create a 2x3 tensor [[1, 2, 3], [4, 0, 6]]
        let input: Tensor<RustGpu, 2, f32> =
            RustGpu::new_tensor(Shape([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0]);

        // Min along dimension 0 (rows)
        let result = input.min(0i32).unwrap();
        let output = result.to_vec();

        assert_eq!(output.len(), 3);
        assert_eq!(output[0], 1.0); // min of [1, 4]
        assert_eq!(output[1], 0.0); // min of [2, 0]
        assert_eq!(output[2], 3.0); // min of [3, 6]
    }

    #[test]
    fn test_min_along_dimension_with_keep_dims() {
        init();

        let input: Tensor<RustGpu, 2, f32> =
            RustGpu::new_tensor(Shape([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0]);

        // Min along dimension 1 (columns) with keep_dims=true
        let result = input.min((1i32, true)).unwrap();
        let shape = result.shape().to_owned();
        let output = result.to_vec();

        assert_eq!(shape, Shape([2, 1])); // Shape preserved with dim=1
        assert_eq!(output.len(), 2);
        assert_eq!(output[0], 1.0); // min of [1, 2, 3]
        assert_eq!(output[1], 0.0); // min of [4, 0, 6]
    }

    #[test]
    fn test_min_negative_values() {
        init();

        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![-3.0, -1.0, -4.0, -2.0]);

        let result = input.min_global().unwrap();
        let output = result.to_vec();

        assert_eq!(output[0], -4.0);
    }

    #[test]
    fn test_min_convenience_methods() {
        init();

        let input: Tensor<RustGpu, 2, f32> =
            RustGpu::new_tensor(Shape([2, 2]), vec![1.0, 2.0, 3.0, 4.0]);

        // Test different parameter formats
        let result1 = input.min(0i32).unwrap(); // From i32
        let result2 = input.min((0i32, false)).unwrap(); // From (i32, bool)
        let result3 = input.min(Some(0i32)).unwrap(); // From Option<i32>

        let out1 = result1.to_vec();
        let out2 = result2.to_vec();
        let out3 = result3.to_vec();

        // All should produce the same result
        assert_eq!(out1, out2);
        assert_eq!(out1, out3);
        assert_eq!(out1[0], 1.0);
        assert_eq!(out1[1], 2.0);
    }
}
