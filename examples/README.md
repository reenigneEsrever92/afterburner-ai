# Afterburner AI Examples

This directory contains examples demonstrating the capabilities of the Afterburner AI framework for GPU-accelerated tensor operations.

## Examples

### Edge Detection (`edge-detection.rs`)

A complete implementation of edge detection using Sobel operators with GPU-accelerated convolution operations.

**Features:**
- Loads and displays images using `egui` GUI framework
- Converts RGB images to grayscale for processing
- Implements Sobel edge detection using Conv2D operations on GPU
- Batch normalization applied to edge detection results
- Side-by-side comparison of original and edge-detected images
- Real-time processing with progress indicators
- Proper error handling and reporting

**Usage:**
```bash
cd examples
cargo run --example edge-detection
```

**Status:** ✅ **COMPLETED AND WORKING**
- Application runs successfully without panics
- CPU-based convolution implementation with padding support
- Interactive GUI with real-time edge detection

**Implementation Details:**
- Uses the Sobel X kernel: `[-1, 0, 1, -2, 0, 2, -1, 0, 1]`
- Creates 4D tensors with shape `[batch, channels, height, width]`
- Applies convolution with stride `[1, 1]` and padding `[1, 1]`
- Batch normalization with configurable gamma and beta parameters
- Padding maintains input image dimensions in output
- CPU backend provides reliable processing for large images
- Adaptive scaling ensures edge visibility across different image types

**Requirements:**
- Image asset: `assets/stones.jpg` (included)
- GPU with compute shader support (for optimal performance)
- Vulkan, DirectX 12, or Metal graphics API support

### YOLOv3 (`yolov3.rs`)

Object detection example using YOLOv3 neural network (implementation may be incomplete).

## Running Examples

Make sure you have the required dependencies installed:

```bash
# Build all examples
cargo build --examples

# Run a specific example
cargo run --example edge-detection
```

## Assets

The `assets/` directory contains sample images and other resources used by the examples:
- `stones.jpg` - Sample image for edge detection

## Architecture

These examples demonstrate the Afterburner AI framework's key components:

1. **Tensor Operations**: Creating and manipulating multi-dimensional tensors
2. **Backend Abstraction**: Seamless switching between GPU (RustGPU) and CPU backends
3. **Conv2D Operations**: GPU-accelerated 2D convolution with configurable stride and padding
4. **Batch Normalization**: Normalizing tensor values with learnable scale and shift parameters
5. **Type Conversions**: Converting between different data types (u8 ↔ f32)
6. **GUI Integration**: Real-time visualization using `egui`

## Development Notes

- The framework uses compute shaders compiled to SPIR-V for GPU operations
- Tensor shapes follow the convention `[batch, channels, height, width]` for 4D tensors
- Conv2D operations support configurable stride and padding parameters
- Batch normalization normalizes activations with learnable gamma and beta parameters
- Padding enables control over output dimensions (same-size with padding=1, or smaller without)
- Error handling includes proper reporting when GPU operations fail
- Examples are designed to be educational and demonstrate best practices

## Troubleshooting

### Edge Detection Runtime Behavior
The edge detection example uses CPU-based processing with padding support:

1. **CPU Processing**: Uses reliable CPU-based Conv2D operations with full padding support
2. **Batch Normalization**: Applies normalization to stabilize edge detection results
3. **Padding Implementation**: Maintains input image dimensions using `padding=[1, 1]`
4. **Adaptive Scaling**: Automatically scales edge values for optimal visibility
5. **Full Image Processing**: Handles large images (tested with 2560×1707 resolution)

**Expected Processing:**
- Converts RGB to grayscale using standard luminance formula
- Applies Sobel edge detection with proper boundary handling
- Scales results automatically for display

### Common Issues
- **Large Images**: CPU processing may take longer for very large images but produces reliable results
- **Processing Time**: Edge detection is compute-intensive; processing indicator shows progress
- **Memory Usage**: Large images require sufficient system RAM for tensor operations

## Next Steps

Potential improvements and extensions:
- Implement full Sobel gradient magnitude calculation (combining X and Y gradients)
- Add more edge detection algorithms (Laplacian, Canny)
- Experiment with different padding strategies (SAME, VALID, custom values)
- Implement batch processing for multiple images
- Add real-time video processing capabilities
- Optimize shader code for better performance

## Conv2D Parameters

The Conv2D operations now support the following parameters:

- **stride**: `Shape<2>` - Controls the step size of the convolution kernel
  - Default: `[1, 1]` (no stride)
  - Example: `[2, 2]` for 2x2 downsampling

- **padding**: `Shape<2>` - Controls zero-padding around input borders
  - Default: `[0, 0]` (no padding)
  - Example: `[1, 1]` to maintain input dimensions with 3x3 kernels

## Batch Normalization Parameters

The Batch Normalization operations support the following parameters:

- **epsilon**: `f32` - Small value added to variance for numerical stability
  - Default: `1e-5`
  - Prevents division by zero when variance is very small

- **gamma**: `Tensor<_, 1, f32>` - Learnable scale parameter per channel
  - Shape: `[num_channels]`
  - Default: `[1.0]` (no scaling)

- **beta**: `Tensor<_, 1, f32>` - Learnable shift parameter per channel
  - Shape: `[num_channels]`
  - Default: `[0.0]` (no shift)

**Batch Normalization Formula**: `output = gamma * (input - mean) / sqrt(variance + epsilon) + beta`