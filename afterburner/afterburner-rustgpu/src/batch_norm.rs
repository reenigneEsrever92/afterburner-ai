use afterburner_core::prelude::*;
use afterburner_ops::prelude::*;
use afterburner_rustgpu_shared::{batch_norm::RustGpuBatchNormParams, RustGpuShape};

use crate::{run_with_backend, RustGpu};

impl BatchNormImpl<RustGpu, f32> for RustGpu {
    fn batch_norm(
        input: &Tensor<RustGpu, 4, f32>,
        gamma: &Tensor<RustGpu, 1, f32>,
        beta: &Tensor<RustGpu, 1, f32>,
        params: BatchNormParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape();

            // Output has same shape as input
            let output = Tensor::create(*input_shape);

            backend.create_buffer(output.id, output.size()).unwrap();

            backend.run_shader_3(
                "batch_norm",
                input.id,
                gamma.id,
                beta.id,
                output.id,
                RustGpuBatchNormParams {
                    dimensions: RustGpuShape(input_shape.0),
                    epsilon: params.epsilon,
                },
            );

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_batch_norm() {
        init();

        // Create a simple 1x2x2x2 tensor
        let input: Tensor<RustGpu, 4, f32> = RustGpu::new_tensor(
            Shape([1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );

        // Gamma (scale) and beta (shift) for 2 channels
        let gamma: Tensor<RustGpu, 1, f32> = RustGpu::new_tensor(Shape([2]), vec![1.0, 1.0]);
        let beta: Tensor<RustGpu, 1, f32> = RustGpu::new_tensor(Shape([2]), vec![0.0, 0.0]);

        let result = input
            .batch_norm(&gamma, &beta, BatchNormParams::default())
            .unwrap();

        let shape = result.shape().to_owned();
        assert_eq!(shape, Shape([1, 2, 2, 2]));

        let output = result.to_vec();

        // Check that the output has reasonable values (normalized)
        assert_eq!(output.len(), 8);

        // The mean of each channel should be approximately 0 after normalization
        let ch0_mean = (output[0] + output[1] + output[2] + output[3]) / 4.0;
        let ch1_mean = (output[4] + output[5] + output[6] + output[7]) / 4.0;

        // Allow some numerical tolerance
        assert!(ch0_mean.abs() < 1e-5);
        assert!(ch1_mean.abs() < 1e-5);
    }
}
