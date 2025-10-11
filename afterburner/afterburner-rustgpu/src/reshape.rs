use afterburner_core::prelude::*;
use afterburner_ops::reshape::{ConcatenateImpl, SliceImpl, TransposeImpl, UpsampleImpl};
use afterburner_rustgpu_shared::reshape::{
    RustGpuConcatenateParams, RustGpuSliceParams, RustGpuTransposeParams, RustGpuUpsampleParams,
};
use afterburner_rustgpu_shared::RustGpuShape;

use crate::{run_with_backend, RustGpu};

impl UpsampleImpl<RustGpu, f32> for RustGpu {
    fn upsample(
        input: &Tensor<RustGpu, 4, f32>,
        params: afterburner_ops::reshape::UpsampleParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape().as_slice();
            let output_shape = [
                input_shape[0],
                input_shape[1],
                input_shape[2] * params.scale_factor,
                input_shape[3] * params.scale_factor,
            ];
            let output = Tensor::create(output_shape);

            let gpu_params = RustGpuUpsampleParams {
                input_shape: RustGpuShape([
                    input_shape[0] as u32,
                    input_shape[1] as u32,
                    input_shape[2] as u32,
                    input_shape[3] as u32,
                ]),
                output_shape: RustGpuShape([
                    output_shape[0] as u32,
                    output_shape[1] as u32,
                    output_shape[2] as u32,
                    output_shape[3] as u32,
                ]),
                scale_factor: params.scale_factor as u32,
                mode: match params.mode {
                    afterburner_ops::reshape::UpsampleMode::Nearest => 0,
                    afterburner_ops::reshape::UpsampleMode::Bilinear => 1,
                },
            };

            let shader_name = match params.mode {
                afterburner_ops::reshape::UpsampleMode::Nearest => "upsample_nearest",
                afterburner_ops::reshape::UpsampleMode::Bilinear => "upsample_bilinear",
            };

            backend.run_shader_with_params(shader_name, input.id, output.id, gpu_params);

            output
        })
    }
}

impl ConcatenateImpl<RustGpu, f32> for RustGpu {
    fn concatenate(
        tensors: &[&Tensor<RustGpu, 4, f32>],
        params: afterburner_ops::reshape::ConcatenateParams,
    ) -> Tensor<RustGpu, 4, f32> {
        if tensors.len() != 2 {
            panic!("Currently only supporting concatenation of 2 tensors");
        }

        run_with_backend(|backend| {
            let a = tensors[0];
            let b = tensors[1];

            let a_shape = a.shape().as_slice();
            let b_shape = b.shape().as_slice();

            let mut output_shape = a_shape.to_vec();
            output_shape[params.axis] += b_shape[params.axis];
            let output = Tensor::create([
                output_shape[0],
                output_shape[1],
                output_shape[2],
                output_shape[3],
            ]);

            let gpu_params = RustGpuConcatenateParams {
                input_a_shape: RustGpuShape([
                    a_shape[0] as u32,
                    a_shape[1] as u32,
                    a_shape[2] as u32,
                    a_shape[3] as u32,
                ]),
                input_b_shape: RustGpuShape([
                    b_shape[0] as u32,
                    b_shape[1] as u32,
                    b_shape[2] as u32,
                    b_shape[3] as u32,
                ]),
                output_shape: RustGpuShape([
                    output_shape[0] as u32,
                    output_shape[1] as u32,
                    output_shape[2] as u32,
                    output_shape[3] as u32,
                ]),
                axis: params.axis as u32,
            };

            backend.run_shader_2("concatenate", a.id, b.id, output.id, gpu_params);

            output
        })
    }
}

impl TransposeImpl<RustGpu, f32> for RustGpu {
    fn transpose(input: &Tensor<RustGpu, 4, f32>, dims: [usize; 4]) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape().as_slice();
            let output_shape = [
                input_shape[dims[0]],
                input_shape[dims[1]],
                input_shape[dims[2]],
                input_shape[dims[3]],
            ];
            let output = Tensor::create(output_shape);

            let gpu_params = RustGpuTransposeParams {
                input_shape: RustGpuShape([
                    input_shape[0] as u32,
                    input_shape[1] as u32,
                    input_shape[2] as u32,
                    input_shape[3] as u32,
                ]),
                output_shape: RustGpuShape([
                    output_shape[0] as u32,
                    output_shape[1] as u32,
                    output_shape[2] as u32,
                    output_shape[3] as u32,
                ]),
                dims: [
                    dims[0] as u32,
                    dims[1] as u32,
                    dims[2] as u32,
                    dims[3] as u32,
                ],
            };

            backend.run_shader_with_params("transpose", input.id, output.id, gpu_params);

            output
        })
    }
}

impl SliceImpl<RustGpu, f32> for RustGpu {
    fn slice(
        input: &Tensor<RustGpu, 4, f32>,
        params: afterburner_ops::reshape::SliceParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create([
                params.size[0],
                params.size[1],
                params.size[2],
                params.size[3],
            ]);

            let input_shape = input.shape().as_slice();
            let gpu_params = RustGpuSliceParams {
                input_shape: RustGpuShape([
                    input_shape[0] as u32,
                    input_shape[1] as u32,
                    input_shape[2] as u32,
                    input_shape[3] as u32,
                ]),
                output_shape: RustGpuShape([
                    params.size[0] as u32,
                    params.size[1] as u32,
                    params.size[2] as u32,
                    params.size[3] as u32,
                ]),
                start: [
                    params.start[0] as u32,
                    params.start[1] as u32,
                    params.start[2] as u32,
                    params.start[3] as u32,
                ],
                size: [
                    params.size[0] as u32,
                    params.size[1] as u32,
                    params.size[2] as u32,
                    params.size[3] as u32,
                ],
            };

            backend.run_shader_with_params("slice", input.id, output.id, gpu_params);

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use afterburner_core::backend::Backend;

    #[test]
    fn test_upsample_nearest() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0], [3.0, 4.0]]]]);

        let result = tensor.upsample(2).unwrap();

        assert_eq!(result.shape().as_slice(), &[1, 1, 4, 4]);
        // Each pixel should be repeated in a 2x2 block
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, // Row 1
            1.0, 1.0, 2.0, 2.0, // Row 2
            3.0, 3.0, 4.0, 4.0, // Row 3
            3.0, 3.0, 4.0, 4.0, // Row 4
        ];
        assert_eq!(result.to_vec(), expected);
    }

    #[test]
    fn test_concatenate_channels() {
        init();
        let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0]], [[3.0, 4.0]]]]);
        let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[5.0, 6.0]], [[7.0, 8.0]]]]);

        let result = a.concatenate(&b, 1).unwrap(); // Concatenate along channel dimension

        assert_eq!(result.shape().as_slice(), &[1, 4, 1, 2]);
        assert_eq!(
            result.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn test_transpose_hwc_to_chw() {
        init();
        // Create tensor in format [batch, height, width, channels]
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0], [2.0]], [[3.0], [4.0]]]]);

        // Transpose to [batch, channels, height, width]
        let result = tensor.transpose([0, 3, 1, 2]).unwrap();

        assert_eq!(result.shape().as_slice(), &[1, 1, 2, 2]);
        assert_eq!(result.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_slice_center() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]]]);

        // Slice 2x2 region from center
        let result = tensor.slice([0, 0, 1, 1], [1, 1, 2, 2]).unwrap();

        assert_eq!(result.shape().as_slice(), &[1, 1, 2, 2]);
        assert_eq!(result.to_vec(), vec![6.0, 7.0, 10.0, 11.0]);
    }
}
