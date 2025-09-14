use afterburner_core::prelude::*;

/// Parameters for channel-wise normalization (torchvision.transforms.Normalize style)
#[derive(Debug, Clone)]
pub struct ChannelNormalizeParams {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Default for ChannelNormalizeParams {
    fn default() -> Self {
        Self {
            mean: vec![],
            std: vec![],
        }
    }
}

/// Backend trait for channel normalization implementation
pub trait ChannelNormalizeImpl<B: Backend, T: Clone> {
    fn channel_normalize(
        input: &Tensor<B, 4, T>,
        params: ChannelNormalizeParams,
    ) -> Tensor<B, 4, T>;
}

/// Convenience trait for channel normalization
pub trait ChannelNormalize<B: Backend, T: Clone> {
    fn channel_normalize(&self, params: ChannelNormalizeParams) -> AbResult<Tensor<B, 4, T>>;
}

impl<B: Backend + ChannelNormalizeImpl<B, T>, T: Clone> ChannelNormalize<B, T> for Tensor<B, 4, T> {
    fn channel_normalize(&self, params: ChannelNormalizeParams) -> AbResult<Tensor<B, 4, T>> {
        let input_shape = self.shape();
        let num_channels = input_shape.as_slice()[1];

        // Validate that mean and std have correct number of channels
        if params.mean.len() != num_channels {
            return Err(Error::ShapeMissmatch);
        }
        if params.std.len() != num_channels {
            return Err(Error::ShapeMissmatch);
        }

        // Validate that std values are not zero
        for &std_val in &params.std {
            if std_val == 0.0 {
                return Err(Error::InvalidParameter);
            }
        }

        Ok(B::channel_normalize(self, params))
    }
}
