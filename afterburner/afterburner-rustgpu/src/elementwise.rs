use afterburner_core::prelude::*;
use afterburner_ops::elementwise::{AddImpl, DivImpl, MulImpl, SubImpl};
use afterburner_rustgpu_shared::elementwise::RustGpuElementwiseParams;

use crate::{run_with_backend, RustGpu};

impl AddImpl<RustGpu, f32> for RustGpu {
    fn add(a: &Tensor<RustGpu, 4, f32>, b: &Tensor<RustGpu, 4, f32>) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create(*a.shape());

            let gpu_params = RustGpuElementwiseParams {
                size: a.shape().size() as u32,
            };

            backend.run_shader_2("elementwise_add", a.id, b.id, output.id, gpu_params);

            output
        })
    }
}

impl SubImpl<RustGpu, f32> for RustGpu {
    fn sub(a: &Tensor<RustGpu, 4, f32>, b: &Tensor<RustGpu, 4, f32>) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create(*a.shape());

            let gpu_params = RustGpuElementwiseParams {
                size: a.shape().size() as u32,
            };

            backend.run_shader_2("elementwise_sub", a.id, b.id, output.id, gpu_params);

            output
        })
    }
}

impl MulImpl<RustGpu, f32> for RustGpu {
    fn mul(a: &Tensor<RustGpu, 4, f32>, b: &Tensor<RustGpu, 4, f32>) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create(*a.shape());

            let gpu_params = RustGpuElementwiseParams {
                size: a.shape().size() as u32,
            };

            backend.run_shader_2("elementwise_mul", a.id, b.id, output.id, gpu_params);

            output
        })
    }
}

impl DivImpl<RustGpu, f32> for RustGpu {
    fn div(a: &Tensor<RustGpu, 4, f32>, b: &Tensor<RustGpu, 4, f32>) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let output = Tensor::create(*a.shape());

            let gpu_params = RustGpuElementwiseParams {
                size: a.shape().size() as u32,
            };

            backend.run_shader_2("elementwise_div", a.id, b.id, output.id, gpu_params);

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use afterburner_core::backend::Backend;

    #[test]
    fn test_elementwise_add() {
        init();
        let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0], [3.0, 4.0]]]]);
        let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 1.0], [1.0, 1.0]]]]);

        let result = a.add(&b).unwrap();

        assert_eq!(result.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_elementwise_sub() {
        init();
        let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[5.0, 6.0], [7.0, 8.0]]]]);
        let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0], [3.0, 4.0]]]]);

        let result = a.sub(&b).unwrap();

        assert_eq!(result.to_vec(), vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_elementwise_mul() {
        init();
        let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[2.0, 3.0], [4.0, 5.0]]]]);
        let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[2.0, 2.0], [2.0, 2.0]]]]);

        let result = a.mul(&b).unwrap();

        assert_eq!(result.to_vec(), vec![4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn test_elementwise_div() {
        init();
        let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[8.0, 6.0], [4.0, 2.0]]]]);
        let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[2.0, 2.0], [2.0, 2.0]]]]);

        let result = a.div(&b).unwrap();

        assert_eq!(result.to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_elementwise_div_zero_protection() {
        init();
        let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0]]]]);
        let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[0.0, 1.0]]]]);

        let result = a.div(&b).unwrap();
        let actual = result.to_vec();

        // First element should not be infinite due to zero protection
        assert!(
            actual[0].is_finite(),
            "Division by zero should be protected"
        );
        assert_eq!(actual[1], 2.0); // 2/1 = 2
    }
}
