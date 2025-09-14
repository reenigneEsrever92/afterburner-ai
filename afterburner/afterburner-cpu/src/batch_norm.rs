use afterburner_core::prelude::*;
use afterburner_ops::prelude::*;

use crate::Cpu;

impl BatchNormImpl<Cpu, f32> for Cpu {
    fn batch_norm(
        input: &Tensor<Cpu, 4, f32>,
        gamma: &Tensor<Cpu, 1, f32>,
        beta: &Tensor<Cpu, 1, f32>,
        params: BatchNormParams,
    ) -> Tensor<Cpu, 4, f32> {
        let input_shape = input.shape();
        let input_data = input.to_vec();
        let gamma_data = gamma.to_vec();
        let beta_data = beta.to_vec();

        // Input dimensions: [batch, channels, height, width]
        let batch_size = input_shape.as_slice()[0];
        let channels = input_shape.as_slice()[1];
        let height = input_shape.as_slice()[2];
        let width = input_shape.as_slice()[3];

        let channel_size = height * width;
        let batch_stride = channels * channel_size;
        let total_elements = input_shape.size();

        let mut output = vec![0.0f32; total_elements];

        // Compute mean and variance for each channel
        let mut channel_means = vec![0.0f32; channels];
        let mut channel_variances = vec![0.0f32; channels];

        for channel in 0..channels {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            let elements_per_channel = batch_size * channel_size;

            // Calculate sum and sum of squares for this channel across all batches
            for batch in 0..batch_size {
                for spatial_idx in 0..channel_size {
                    let idx = batch * batch_stride + channel * channel_size + spatial_idx;
                    let val = input_data[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }

            let mean = sum / elements_per_channel as f32;
            let variance = (sum_sq / elements_per_channel as f32) - (mean * mean);

            channel_means[channel] = mean;
            channel_variances[channel] = variance;
        }

        // Apply normalization to each element
        for batch in 0..batch_size {
            for channel in 0..channels {
                let mean = channel_means[channel];
                let variance = channel_variances[channel];
                let std_dev = (variance + params.epsilon).sqrt();
                let scale = gamma_data[channel];
                let shift = beta_data[channel];

                for spatial_idx in 0..channel_size {
                    let idx = batch * batch_stride + channel * channel_size + spatial_idx;
                    let input_val = input_data[idx];
                    let normalized = (input_val - mean) / std_dev;
                    let scaled_shifted = normalized * scale + shift;
                    output[idx] = scaled_shifted;
                }
            }
        }

        Cpu::new_tensor(*input_shape, output)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_batch_norm() {
        // Create a simple 1x2x2x2 tensor
        let input: Tensor<Cpu, 4, f32> = Cpu::new_tensor(
            Shape([1, 2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );

        // Gamma (scale) and beta (shift) for 2 channels
        let gamma: Tensor<Cpu, 1, f32> = Cpu::new_tensor(Shape([2]), vec![1.0, 1.0]);
        let beta: Tensor<Cpu, 1, f32> = Cpu::new_tensor(Shape([2]), vec![0.0, 0.0]);

        let result = input
            .batch_norm(&gamma, &beta, BatchNormParams::default())
            .unwrap();

        let shape = result.shape().to_owned();
        assert_eq!(shape, Shape([1, 2, 2, 2]));

        let output = result.to_vec();

        // Check that the output has correct dimensions
        assert_eq!(output.len(), 8);

        // The mean of each channel should be approximately 0 after normalization
        let ch0_mean = (output[0] + output[1] + output[2] + output[3]) / 4.0;
        let ch1_mean = (output[4] + output[5] + output[6] + output[7]) / 4.0;

        // Allow some numerical tolerance
        assert!(ch0_mean.abs() < 1e-5);
        assert!(ch1_mean.abs() < 1e-5);
    }

    #[test]
    fn test_batch_norm_with_scale_shift() {
        // Single batch, single channel test
        let input: Tensor<Cpu, 4, f32> =
            Cpu::new_tensor(Shape([1, 1, 2, 2]), vec![2.0, 4.0, 6.0, 8.0]);

        let gamma: Tensor<Cpu, 1, f32> = Cpu::new_tensor(Shape([1]), vec![2.0]);
        let beta: Tensor<Cpu, 1, f32> = Cpu::new_tensor(Shape([1]), vec![1.0]);

        let result = input
            .batch_norm(&gamma, &beta, BatchNormParams::default())
            .unwrap();

        let output = result.to_vec();

        // After normalization, mean should be 1.0 (the beta value) and variance should be scaled by gamma^2
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        assert!((mean - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_batch_norm_multi_batch() {
        // Multiple batches test
        let input: Tensor<Cpu, 4, f32> =
            Cpu::new_tensor(Shape([2, 1, 1, 2]), vec![1.0, 2.0, 3.0, 4.0]);

        let gamma: Tensor<Cpu, 1, f32> = Cpu::new_tensor(Shape([1]), vec![1.0]);
        let beta: Tensor<Cpu, 1, f32> = Cpu::new_tensor(Shape([1]), vec![0.0]);

        let result = input
            .batch_norm(&gamma, &beta, BatchNormParams::default())
            .unwrap();

        let output = result.to_vec();

        // Check overall normalization across batches
        let mean = output.iter().sum::<f32>() / output.len() as f32;
        assert!(mean.abs() < 1e-5);
    }
}
