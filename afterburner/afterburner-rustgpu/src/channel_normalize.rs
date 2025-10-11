use afterburner_core::prelude::*;
use afterburner_ops::prelude::*;
use afterburner_rustgpu_shared::{channel_normalize::RustGpuChannelNormalizeParams, RustGpuShape};

use crate::{run_with_backend, RustGpu};

impl ChannelNormalizeImpl<RustGpu, f32> for RustGpu {
    fn channel_normalize(
        input: &Tensor<RustGpu, 4, f32>,
        params: ChannelNormalizeParams,
    ) -> Tensor<RustGpu, 4, f32> {
        run_with_backend(|backend| {
            let input_shape = input.shape();

            // Output has same shape as input
            let output = Tensor::create(*input_shape);

            backend.create_buffer(output.id, output.size()).unwrap();

            let gpu_params = RustGpuChannelNormalizeParams::new(
                RustGpuShape([
                    input_shape.0[0] as u32,
                    input_shape.0[1] as u32,
                    input_shape.0[2] as u32,
                    input_shape.0[3] as u32,
                ]),
                &params.mean,
                &params.std,
            );

            backend.run_shader_with_params("channel_normalize", input.id, output.id, gpu_params);

            output
        })
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;
    use afterburner_core::backend::Backend;

    #[test]
    fn test_channel_normalize_rgb() {
        init();

        // Create a 1x3x2x2 tensor (1 batch, 3 RGB channels, 2x2 spatial)
        let input: Tensor<RustGpu, 4, f32> = RustGpu::new_tensor(
            Shape([1, 3, 2, 2]),
            vec![
                // Channel 0 (R): [100, 150, 200, 250]
                100.0, 150.0, 200.0, 250.0, // Channel 1 (G): [50, 100, 150, 200]
                50.0, 100.0, 150.0, 200.0, // Channel 2 (B): [75, 125, 175, 225]
                75.0, 125.0, 175.0, 225.0,
            ],
        );

        // Use typical ImageNet normalization values (scaled for easier testing)
        let mean = vec![100.0, 100.0, 100.0]; // Mean values for RGB channels
        let std = vec![50.0, 50.0, 50.0]; // Std values for RGB channels

        let result = input
            .channel_normalize(ChannelNormalizeParams { mean, std })
            .unwrap();

        let output = result.to_vec();
        assert_eq!(output.len(), 12);

        // Channel 0: (input - 100.0) / 50.0
        // [100, 150, 200, 250] -> [0, 1, 2, 3]
        assert!((output[0] - 0.0).abs() < 1e-5);
        assert!((output[1] - 1.0).abs() < 1e-5);
        assert!((output[2] - 2.0).abs() < 1e-5);
        assert!((output[3] - 3.0).abs() < 1e-5);

        // Channel 1: (input - 100.0) / 50.0
        // [50, 100, 150, 200] -> [-1, 0, 1, 2]
        assert!((output[4] - (-1.0)).abs() < 1e-5);
        assert!((output[5] - 0.0).abs() < 1e-5);
        assert!((output[6] - 1.0).abs() < 1e-5);
        assert!((output[7] - 2.0).abs() < 1e-5);

        // Channel 2: (input - 100.0) / 50.0
        // [75, 125, 175, 225] -> [-0.5, 0.5, 1.5, 2.5]
        assert!((output[8] - (-0.5)).abs() < 1e-5);
        assert!((output[9] - 0.5).abs() < 1e-5);
        assert!((output[10] - 1.5).abs() < 1e-5);
        assert!((output[11] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_channel_normalize_imagenet() {
        init();

        // Create a small RGB image tensor
        let input: Tensor<RustGpu, 4, f32> = RustGpu::new_tensor(
            Shape([1, 3, 2, 2]),
            vec![
                // Simulate normalized RGB values in [0, 1] range
                0.5, 0.6, 0.7, 0.8, // R channel
                0.4, 0.5, 0.6, 0.7, // G channel
                0.3, 0.4, 0.5, 0.6, // B channel
            ],
        );

        // Standard ImageNet normalization parameters
        let imagenet_mean = vec![0.485, 0.456, 0.406];
        let imagenet_std = vec![0.229, 0.224, 0.225];

        let result = input
            .channel_normalize(ChannelNormalizeParams {
                mean: imagenet_mean,
                std: imagenet_std,
            })
            .unwrap();

        let output = result.to_vec();
        assert_eq!(output.len(), 12);

        // Verify the formula: (input - mean) / std
        // For first pixel of each channel:
        // R: (0.5 - 0.485) / 0.229 ≈ 0.0656
        // G: (0.4 - 0.456) / 0.224 ≈ -0.25
        // B: (0.3 - 0.406) / 0.225 ≈ -0.471

        assert!((output[0] - 0.0656).abs() < 1e-2);
        assert!((output[4] - (-0.25)).abs() < 1e-2);
        assert!((output[8] - (-0.471)).abs() < 1e-2);
    }

    #[test]
    fn test_channel_normalize_batch() {
        init();

        // Create a 2x2x2x2 tensor (2 batches, 2 channels, 2x2 spatial)
        let input: Tensor<RustGpu, 4, f32> = RustGpu::new_tensor(
            Shape([2, 2, 2, 2]),
            vec![
                // Batch 0, Channel 0: [1, 2, 3, 4]
                1.0, 2.0, 3.0, 4.0, // Batch 0, Channel 1: [5, 6, 7, 8]
                5.0, 6.0, 7.0, 8.0, // Batch 1, Channel 0: [9, 10, 11, 12]
                9.0, 10.0, 11.0, 12.0, // Batch 1, Channel 1: [13, 14, 15, 16]
                13.0, 14.0, 15.0, 16.0,
            ],
        );

        let mean = vec![2.0, 6.0]; // Channel means
        let std = vec![1.0, 2.0]; // Channel stds

        let result = input
            .channel_normalize(ChannelNormalizeParams { mean, std })
            .unwrap();

        let output = result.to_vec();
        assert_eq!(output.len(), 16);

        // Batch 0, Channel 0: (input - 2.0) / 1.0
        // [1, 2, 3, 4] -> [-1, 0, 1, 2]
        assert!((output[0] - (-1.0)).abs() < 1e-5);
        assert!((output[1] - 0.0).abs() < 1e-5);
        assert!((output[2] - 1.0).abs() < 1e-5);
        assert!((output[3] - 2.0).abs() < 1e-5);

        // Batch 0, Channel 1: (input - 6.0) / 2.0
        // [5, 6, 7, 8] -> [-0.5, 0, 0.5, 1]
        assert!((output[4] - (-0.5)).abs() < 1e-5);
        assert!((output[5] - 0.0).abs() < 1e-5);
        assert!((output[6] - 0.5).abs() < 1e-5);
        assert!((output[7] - 1.0).abs() < 1e-5);

        // Batch 1, Channel 0: (input - 2.0) / 1.0
        // [9, 10, 11, 12] -> [7, 8, 9, 10]
        assert!((output[8] - 7.0).abs() < 1e-5);
        assert!((output[9] - 8.0).abs() < 1e-5);
        assert!((output[10] - 9.0).abs() < 1e-5);
        assert!((output[11] - 10.0).abs() < 1e-5);

        // Batch 1, Channel 1: (input - 6.0) / 2.0
        // [13, 14, 15, 16] -> [3.5, 4, 4.5, 5]
        assert!((output[12] - 3.5).abs() < 1e-5);
        assert!((output[13] - 4.0).abs() < 1e-5);
        assert!((output[14] - 4.5).abs() < 1e-5);
        assert!((output[15] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_channel_normalize_convenience() {
        init();

        let input: Tensor<RustGpu, 4, f32> =
            RustGpu::new_tensor(Shape([1, 2, 1, 1]), vec![10.0, 20.0]);

        // Test direct struct creation
        let result1 = input
            .channel_normalize(ChannelNormalizeParams {
                mean: vec![5.0, 10.0],
                std: vec![2.0, 4.0],
            })
            .unwrap();
        let result2 = input
            .channel_normalize(ChannelNormalizeParams {
                mean: vec![5.0, 10.0],
                std: vec![2.0, 4.0],
            })
            .unwrap();

        let out1 = result1.to_vec();
        let out2 = result2.to_vec();

        // Both should produce the same result
        assert!((out1[0] - out2[0]).abs() < 1e-6);
        assert!((out1[1] - out2[1]).abs() < 1e-6);

        // Channel 0: (10 - 5) / 2 = 2.5
        // Channel 1: (20 - 10) / 4 = 2.5
        assert!((out1[0] - 2.5).abs() < 1e-5);
        assert!((out1[1] - 2.5).abs() < 1e-5);
    }
}
