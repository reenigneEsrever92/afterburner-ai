use afterburner_core::prelude::*;
use afterburner_ops::activation::SigmoidImpl;
use afterburner_rustgpu_shared::activation::RustGpuSigmoidParams;

use crate::{run_with_backend, RustGpu};

impl SigmoidImpl<RustGpu, f32> for RustGpu {
    fn sigmoid(input: &Tensor<RustGpu, 4, f32>) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create(*input.shape());

            let gpu_params = RustGpuSigmoidParams {
                size: input.shape().size() as u32,
            };

            backend.run_shader_with_params("sigmoid", input.id, output.id, gpu_params);

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use afterburner_core::backend::Backend;

    #[test]
    fn test_sigmoid_zero() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[0.0]]]]);
        let result = tensor.sigmoid().unwrap();

        // sigmoid(0) = 0.5
        let expected = 0.5;
        let actual = result.to_vec()[0];
        assert!(
            (actual - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_sigmoid_positive() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0]]]]);
        let result = tensor.sigmoid().unwrap();

        let actual = result.to_vec();
        // sigmoid(1) ≈ 0.7311, sigmoid(2) ≈ 0.8808
        assert!(
            actual[0] > 0.7 && actual[0] < 0.8,
            "sigmoid(1) should be ~0.731, got {}",
            actual[0]
        );
        assert!(
            actual[1] > 0.8 && actual[1] < 0.9,
            "sigmoid(2) should be ~0.881, got {}",
            actual[1]
        );
    }

    #[test]
    fn test_sigmoid_negative() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[-1.0, -2.0]]]]);
        let result = tensor.sigmoid().unwrap();

        let actual = result.to_vec();
        // sigmoid(-1) ≈ 0.2689, sigmoid(-2) ≈ 0.1192
        assert!(
            actual[0] > 0.2 && actual[0] < 0.3,
            "sigmoid(-1) should be ~0.269, got {}",
            actual[0]
        );
        assert!(
            actual[1] > 0.1 && actual[1] < 0.2,
            "sigmoid(-2) should be ~0.119, got {}",
            actual[1]
        );
    }

    #[test]
    fn test_sigmoid_large_values() {
        init();
        let tensor: Tensor<RustGpu, 4, f32> = Tensor::from([[[[100.0, -100.0]]]]);
        let result = tensor.sigmoid().unwrap();

        let actual = result.to_vec();
        // sigmoid(100) ≈ 1.0, sigmoid(-100) ≈ 0.0
        assert!(
            actual[0] > 0.99,
            "sigmoid(100) should be ~1.0, got {}",
            actual[0]
        );
        assert!(
            actual[1] < 0.01,
            "sigmoid(-100) should be ~0.0, got {}",
            actual[1]
        );
    }
}
