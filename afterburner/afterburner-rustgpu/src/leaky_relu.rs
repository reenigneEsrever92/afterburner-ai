use afterburner_core::prelude::*;
use afterburner_ops::activation::{LeakyReLUImpl, LeakyReLUParams};
use afterburner_rustgpu_shared::activation::RustGpuLeakyReLUParams;

use crate::{run_with_backend, RustGpu};

impl LeakyReLUImpl<RustGpu, f32> for RustGpu {
    fn leaky_relu(
        input: &Tensor<RustGpu, 4, f32>,
        params: LeakyReLUParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create(*input.shape());

            let gpu_params = RustGpuLeakyReLUParams {
                alpha: params.alpha,
                size: input.shape().size() as u32,
            };

            backend.run_shader_with_params("leaky_relu", input.id, output.id, gpu_params);

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use afterburner_core::backend::Backend;

    #[test]
    fn test_leaky_relu_positive() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[2.0, 3.0], [1.0, 4.0]]]]);
        let result = tensor.leaky_relu(0.1).unwrap();

        // Positive values should remain unchanged
        assert_eq!(result.to_vec(), vec![2.0, 3.0, 1.0, 4.0]);
    }

    #[test]
    fn test_leaky_relu_negative() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[-2.0, -3.0], [-1.0, -4.0]]]]);
        let result = tensor.leaky_relu(0.1).unwrap();

        // Negative values should be multiplied by alpha
        let expected = vec![-0.2, -0.3, -0.1, -0.4];
        let actual = result.to_vec();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }
    }

    #[test]
    fn test_leaky_relu_mixed() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[2.0, -1.0], [0.0, -3.0]]]]);
        let result = tensor.leaky_relu(0.2).unwrap();

        let expected = vec![2.0, -0.2, 0.0, -0.6];
        let actual = result.to_vec();

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-6, "Expected {}, got {}", e, a);
        }
    }
}
