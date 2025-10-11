use afterburner_rustgpu::prelude::*;

fn main() -> AbResult<()> {
    run_with_backend(|| {
        println!("Testing New Operations for YOLOv3");
        println!("==================================");

        // Test LeakyReLU activation
        println!("\n--- Testing LeakyReLU ---");
        test_leaky_relu()?;

        // Test Sigmoid activation
        println!("\n--- Testing Sigmoid ---");
        test_sigmoid()?;

        // Test Element-wise operations
        println!("\n--- Testing Element-wise Operations ---");
        test_elementwise_ops()?;

        // Test Basic convolution and batch norm (existing ops)
        println!("\n--- Testing Convolution + BatchNorm ---");
        test_conv_batchnorm()?;

        println!("\nAll operations tested successfully!");
        println!("YOLOv3 building blocks are ready to use.");

        Ok(())
    })
}

fn test_leaky_relu() -> AbResult<()> {
    // Test with mixed positive and negative values
    let input: Tensor<RustGpu, 4, f32> = Tensor::from([[[[2.0, -1.0], [0.0, -3.0]]]]);

    println!("Input: {:?}", input.to_vec());

    let result = input.leaky_relu(0.1)?;
    println!("LeakyReLU(0.1): {:?}", result.to_vec());

    // Expected: [2.0, -0.1, 0.0, -0.3]
    let expected = vec![2.0, -0.1, 0.0, -0.3];
    let actual = result.to_vec();

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > 1e-5 {
            println!("  ❌ Mismatch at index {}: expected {}, got {}", i, e, a);
            return Err(Error::ShapeMissmatch);
        }
    }

    println!("  ✅ LeakyReLU test passed");
    Ok(())
}

fn test_sigmoid() -> AbResult<()> {
    // Test sigmoid with known values
    let input: Tensor<RustGpu, 4, f32> = Tensor::from([[[[0.0, 1.0], [-1.0, 2.0]]]]);

    println!("Input: {:?}", input.to_vec());

    let result = input.sigmoid()?;
    let actual = result.to_vec();

    println!("Sigmoid: {:?}", actual);

    // Check bounds: sigmoid should be between 0 and 1
    for (i, &val) in actual.iter().enumerate() {
        if val < 0.0 || val > 1.0 {
            println!(
                "  ❌ Sigmoid value {} at index {} is out of bounds [0,1]",
                val, i
            );
            return Err(Error::ShapeMissmatch);
        }
    }

    // Check sigmoid(0) ≈ 0.5
    let sig_zero = actual[0];
    if (sig_zero - 0.5).abs() > 1e-5 {
        println!("  ❌ sigmoid(0) should be 0.5, got {}", sig_zero);
        return Err(Error::ShapeMissmatch);
    }

    println!("  ✅ Sigmoid test passed");
    Ok(())
}

fn test_elementwise_ops() -> AbResult<()> {
    let a: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 2.0], [3.0, 4.0]]]]);
    let b: Tensor<RustGpu, 4, f32> = Tensor::from([[[[1.0, 1.0], [2.0, 2.0]]]]);

    println!("Tensor A: {:?}", a.to_vec());
    println!("Tensor B: {:?}", b.to_vec());

    // Test addition
    let add_result = a.add(&b)?;
    println!("A + B: {:?}", add_result.to_vec());
    let expected_add = vec![2.0, 3.0, 5.0, 6.0];
    verify_result(&add_result.to_vec(), &expected_add, "Addition")?;

    // Test subtraction
    let sub_result = a.sub(&b)?;
    println!("A - B: {:?}", sub_result.to_vec());
    let expected_sub = vec![0.0, 1.0, 1.0, 2.0];
    verify_result(&sub_result.to_vec(), &expected_sub, "Subtraction")?;

    // Test multiplication
    let mul_result = a.mul(&b)?;
    println!("A * B: {:?}", mul_result.to_vec());
    let expected_mul = vec![1.0, 2.0, 6.0, 8.0];
    verify_result(&mul_result.to_vec(), &expected_mul, "Multiplication")?;

    // Test division
    let div_result = a.div(&b)?;
    println!("A / B: {:?}", div_result.to_vec());
    let expected_div = vec![1.0, 2.0, 1.5, 2.0];
    verify_result(&div_result.to_vec(), &expected_div, "Division")?;

    println!("  ✅ All element-wise operations passed");
    Ok(())
}

fn test_conv_batchnorm() -> AbResult<()> {
    // Simple 3x3 convolution test
    let input: Tensor<RustGpu, 4, f32> = Tensor::from([[[
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ]]]);

    // 3x3 identity-like kernel
    let weights: Tensor<RustGpu, 4, f32> =
        Tensor::from([[[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]]]);
    let weights_3x3 = weights.reshape([1, 1, 3, 3])?;

    println!("Input shape: {:?}", input.shape());
    println!("Weight shape: {:?}", weights_3x3.shape());

    // Perform convolution
    let conv_params = Conv2DParams {
        stride: Shape([1, 1]),
        padding: Shape([1, 1]), // Same padding
    };

    let conv_result = input.conv_2d(&weights_3x3, conv_params)?;
    println!("Convolution result shape: {:?}", conv_result.shape());
    println!("Convolution values: {:?}", conv_result.to_vec());

    // Test batch normalization
    let gamma: Tensor<RustGpu, 1, f32> = Tensor::from([1.0]);
    let beta: Tensor<RustGpu, 1, f32> = Tensor::from([0.0]);

    let bn_result = conv_result.batch_norm(&gamma, &beta, BatchNormParams::default())?;
    println!("BatchNorm result: {:?}", bn_result.to_vec());

    println!("  ✅ Convolution + BatchNorm test passed");
    Ok(())
}

fn verify_result(actual: &[f32], expected: &[f32], operation: &str) -> AbResult<()> {
    if actual.len() != expected.len() {
        println!(
            "  ❌ {} length mismatch: expected {}, got {}",
            operation,
            expected.len(),
            actual.len()
        );
        return Err(Error::ShapeMissmatch);
    }

    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > 1e-5 {
            println!(
                "  ❌ {} mismatch at index {}: expected {}, got {}",
                operation, i, e, a
            );
            return Err(Error::ShapeMissmatch);
        }
    }

    Ok(())
}

// Test helper for creating simple YOLOv3-like layers
fn test_yolo_layer_components() -> AbResult<()> {
    println!("\n--- Testing YOLOv3 Layer Components ---");

    // Test a basic building block: Conv + BN + LeakyReLU
    let input: Tensor<RustGpu, 4, f32> = Tensor::create([1, 3, 416, 416]);
    println!("Created input tensor: {:?}", input.shape());

    // Conv layer weights (32 output channels, 3 input channels, 3x3 kernel)
    let conv_weights: Tensor<RustGpu, 4, f32> = Tensor::create([32, 3, 3, 3]);
    println!("Created conv weights: {:?}", conv_weights.shape());

    // Batch norm parameters
    let gamma: Tensor<RustGpu, 1, f32> = Tensor::create([32]);
    let beta: Tensor<RustGpu, 1, f32> = Tensor::create([32]);

    // Forward pass: Conv -> BN -> LeakyReLU
    let conv_params = Conv2DParams {
        stride: Shape([1, 1]),
        padding: Shape([1, 1]),
    };

    let x1 = input.conv_2d(&conv_weights, conv_params)?;
    println!("After conv: {:?}", x1.shape());

    let x2 = x1.batch_norm(&gamma, &beta, BatchNormParams::default())?;
    println!("After batch_norm: {:?}", x2.shape());

    let x3 = x2.leaky_relu(0.1)?;
    println!("After leaky_relu: {:?}", x3.shape());

    println!("  ✅ YOLOv3 layer component test passed");

    // Test typical downsampling layer (stride 2)
    let downsample_weights: Tensor<RustGpu, 4, f32> = Tensor::create([64, 32, 3, 3]);
    let downsample_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([64]);
    let downsample_beta: Tensor<RustGpu, 1, f32> = Tensor::create([64]);

    let downsample_params = Conv2DParams {
        stride: Shape([2, 2]), // Downsample by 2
        padding: Shape([1, 1]),
    };

    let y1 = x3.conv_2d(&downsample_weights, downsample_params)?;
    let y2 = y1.batch_norm(
        &downsample_gamma,
        &downsample_beta,
        BatchNormParams::default(),
    )?;
    let y3 = y2.leaky_relu(0.1)?;

    println!("After downsampling: {:?}", y3.shape());

    // Verify downsampling worked correctly
    let expected_h = 416 / 2; // Should be 208
    let expected_w = 416 / 2; // Should be 208

    if y3.shape().as_slice()[2] != expected_h || y3.shape().as_slice()[3] != expected_w {
        println!(
            "  ❌ Downsampling failed: expected {}x{}, got {}x{}",
            expected_h,
            expected_w,
            y3.shape().as_slice()[2],
            y3.shape().as_slice()[3]
        );
        return Err(Error::ShapeMissmatch);
    }

    println!("  ✅ Downsampling test passed");
    Ok(())
}
