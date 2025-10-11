use afterburner_core::prelude::*;
use afterburner_ops::transforms::range_normalize::{RangeNormalizeImpl, RangeNormalizeParams};

use crate::{run_with_backend, RustGpu};

impl<const D: usize> RangeNormalizeImpl<RustGpu, f32, D> for RustGpu {
    fn range_normalize(
        input: &Tensor<RustGpu, D, f32>,
        params: RangeNormalizeParams,
    ) -> Tensor<RustGpu, D, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape();

            // Output has same shape as input
            let output = Tensor::create(*input_shape);
            backend.create_buffer(output.id, output.size()).unwrap();

            // Calculate total size
            let total_size = input_shape.size();

            let gpu_params =
                afterburner_rustgpu_shared::range_normalize::RustGpuRangeNormalizeParams::new(
                    total_size, params.eps,
                );

            backend.run_shader_with_params("range_normalize", input.id, output.id, gpu_params);

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{init, RustGpu};
    use crate::{Shape, Tensor};
    use afterburner_ops::transforms::range_normalize::{RangeNormalize, RangeNormalizeParams};

    #[test]
    fn test_range_normalize_1d() {
        init();

        // Create a 1D tensor [2, 4, 6, 8]
        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![2.0, 4.0, 6.0, 8.0]);

        let result = input
            .range_normalize(RangeNormalizeParams::default())
            .unwrap();
        let output = result.to_vec();

        // Expected: (value - 2) / (8 - 2) = (value - 2) / 6
        // [2, 4, 6, 8] -> [0, 1/3, 2/3, 1]
        assert_eq!(output.len(), 4);
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((output[2] - 2.0 / 3.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_range_normalize_2d() {
        init();

        // Create a 2x2 tensor [[1, 3], [5, 7]]
        let input: Tensor<RustGpu, 2, f32> =
            RustGpu::new_tensor(Shape([2, 2]), vec![1.0, 3.0, 5.0, 7.0]);

        let result = input.range_normalize_default().unwrap();
        let output = result.to_vec();

        // Expected: (value - 1) / (7 - 1) = (value - 1) / 6
        // [1, 3, 5, 7] -> [0, 1/3, 2/3, 1]
        assert_eq!(output.len(), 4);
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((output[2] - 2.0 / 3.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_range_normalize_same_values() {
        init();

        // Create a tensor with all same values
        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![5.0, 5.0, 5.0, 5.0]);

        let result = input
            .range_normalize(RangeNormalizeParams::from(1e-6f32))
            .unwrap();
        let output = result.to_vec();

        // When all values are the same, they should all become 0
        for &val in &output {
            assert!((val - 0.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_range_normalize_negative_values() {
        init();

        // Test with negative values
        let input: Tensor<RustGpu, 1, f32> =
            RustGpu::new_tensor(Shape([4]), vec![-4.0, -2.0, 0.0, 2.0]);

        let result = input.range_normalize_default().unwrap();
        let output = result.to_vec();

        // Expected: (value - (-4)) / (2 - (-4)) = (value + 4) / 6
        // [-4, -2, 0, 2] -> [0, 1/3, 2/3, 1]
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 1.0 / 3.0).abs() < 1e-5);
        assert!((output[2] - 2.0 / 3.0).abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }
}
