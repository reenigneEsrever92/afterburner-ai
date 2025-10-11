use afterburner_rustgpu::prelude::*;

fn main() -> AbResult<()> {
    init();
    println!("YOLOv3 Object Detection Example");
    println!("================================");

    // Create input tensor (batch=1, channels=3, height=416, width=416)
    let input = create_dummy_image();
    println!("Input shape: {:?}", input.shape());

    // YOLOv3 Backbone Feature Extraction
    println!("\n--- Feature Extraction ---");

    // Initial conv layer: 3 -> 32 channels
    let conv1_weights: Tensor<RustGpu, 4, f32> = Tensor::create([32, 3, 3, 3]);
    let conv1_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([32]);
    let conv1_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([32]);

    let x1 = input.conv_2d(
        &conv1_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]),
        },
    )?;
    let x1 = x1.batch_norm(&conv1_bn_gamma, &conv1_bn_beta, BatchNormParams::default())?;
    let x1 = leaky_relu(&x1, 0.1)?;
    println!("After conv1: {:?}", x1.shape());

    // Downsample 1: 32 -> 64 channels, stride 2
    let conv2_weights: Tensor<RustGpu, 4, f32> = Tensor::create([64, 32, 3, 3]);
    let conv2_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([64]);
    let conv2_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([64]);

    let x2 = x1.conv_2d(
        &conv2_weights,
        Conv2DParams {
            stride: Shape([2, 2]),
            padding: Shape([1, 1]),
        },
    )?;
    let x2 = x2.batch_norm(&conv2_bn_gamma, &conv2_bn_beta, BatchNormParams::default())?;
    let x2 = leaky_relu(&x2, 0.1)?;
    println!("After downsample1: {:?}", x2.shape());

    // Residual blocks (simplified)
    let x2 = residual_block(&x2, 64)?;

    // Downsample 2: 64 -> 128 channels
    let conv3_weights: Tensor<RustGpu, 4, f32> = Tensor::create([128, 64, 3, 3]);
    let conv3_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([128]);
    let conv3_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([128]);

    let x3 = x2.conv_2d(
        &conv3_weights,
        Conv2DParams {
            stride: Shape([2, 2]),
            padding: Shape([1, 1]),
        },
    )?;
    let x3 = x3.batch_norm(&conv3_bn_gamma, &conv3_bn_beta, BatchNormParams::default())?;
    let x3 = leaky_relu(&x3, 0.1)?;
    let x3 = residual_block(&x3, 128)?;
    let x3 = residual_block(&x3, 128)?;
    println!("After downsample2: {:?}", x3.shape());

    // Downsample 3: 128 -> 256 channels
    let conv4_weights: Tensor<RustGpu, 4, f32> = Tensor::create([256, 128, 3, 3]);
    let conv4_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([256]);
    let conv4_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([256]);

    let x4 = x3.conv_2d(
        &conv4_weights,
        Conv2DParams {
            stride: Shape([2, 2]),
            padding: Shape([1, 1]),
        },
    )?;
    let x4 = x4.batch_norm(&conv4_bn_gamma, &conv4_bn_beta, BatchNormParams::default())?;
    let x4 = leaky_relu(&x4, 0.1)?;

    // Multiple residual blocks
    let mut x4_res = x4;
    for _ in 0..8 {
        x4_res = residual_block(&x4_res, 256)?;
    }
    let feature_52x52 = x4_res.clone(); // Save for skip connection
    println!("Feature 52x52: {:?}", feature_52x52.shape());

    // Downsample 4: 256 -> 512 channels
    let conv5_weights: Tensor<RustGpu, 4, f32> = Tensor::create([512, 256, 3, 3]);
    let conv5_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([512]);
    let conv5_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([512]);

    let x5 = x4_res.conv_2d(
        &conv5_weights,
        Conv2DParams {
            stride: Shape([2, 2]),
            padding: Shape([1, 1]),
        },
    )?;
    let x5 = x5.batch_norm(&conv5_bn_gamma, &conv5_bn_beta, BatchNormParams::default())?;
    let x5 = leaky_relu(&x5, 0.1)?;

    let mut x5_res = x5;
    for _ in 0..8 {
        x5_res = residual_block(&x5_res, 512)?;
    }
    let feature_26x26 = x5_res.clone(); // Save for skip connection
    println!("Feature 26x26: {:?}", feature_26x26.shape());

    // Downsample 5: 512 -> 1024 channels
    let conv6_weights: Tensor<RustGpu, 4, f32> = Tensor::create([1024, 512, 3, 3]);
    let conv6_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([1024]);
    let conv6_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([1024]);

    let x6 = x5_res.conv_2d(
        &conv6_weights,
        Conv2DParams {
            stride: Shape([2, 2]),
            padding: Shape([1, 1]),
        },
    )?;
    let x6 = x6.batch_norm(&conv6_bn_gamma, &conv6_bn_beta, BatchNormParams::default())?;
    let x6 = leaky_relu(&x6, 0.1)?;

    let mut x6_res = x6;
    for _ in 0..4 {
        x6_res = residual_block(&x6_res, 1024)?;
    }
    let feature_13x13 = x6_res;
    println!("Feature 13x13: {:?}", feature_13x13.shape());

    // YOLOv3 Detection Heads
    println!("\n--- Detection Heads ---");

    // Scale 1: 13x13 detection
    let det1 = detection_head(&feature_13x13, 1024, 255)?; // 255 = 3*(5+80) for COCO
    println!("Detection 13x13: {:?}", det1.shape());

    // Scale 2: 26x26 detection (with upsampling and skip connection)
    let up1_weights: Tensor<RustGpu, 4, f32> = Tensor::create([512, 1024, 1, 1]);
    let up1 = feature_13x13.conv_2d(
        &up1_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([0, 0]),
        },
    )?;
    let up1 = upsample(&up1, 2)?; // 13x13 -> 26x26
    let concat1 = concatenate(&up1, &feature_26x26, 1)?; // Concat along channel dim
    let det2 = detection_head(&concat1, 768, 255)?; // 512 + 256 = 768 input channels
    println!("Detection 26x26: {:?}", det2.shape());

    // Scale 3: 52x52 detection (with upsampling and skip connection)
    let up2_weights: Tensor<RustGpu, 4, f32> = Tensor::create([256, 768, 1, 1]);
    let up2 = concat1.conv_2d(
        &up2_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([0, 0]),
        },
    )?;
    let up2 = upsample(&up2, 2)?; // 26x26 -> 52x52
    let concat2 = concatenate(&up2, &feature_52x52, 1)?; // Concat along channel dim
    let det3 = detection_head(&concat2, 512, 255)?; // 256 + 256 = 512 input channels
    println!("Detection 52x52: {:?}", det3.shape());

    // Post-processing
    println!("\n--- Post-processing ---");

    let predictions = vec![det1, det2, det3];
    let detections = decode_yolo_outputs(&predictions)?;
    let final_detections = apply_nms(&detections, 0.4)?;

    println!("Total raw detections: {}", detections.len());
    println!("Final detections after NMS: {}", final_detections.len());

    // Print some sample detections
    for (i, det) in final_detections.iter().take(5).enumerate() {
        println!(
            "Detection {}: bbox=[{:.1}, {:.1}, {:.1}, {:.1}], conf={:.3}, class={}",
            i, det.0[0], det.0[1], det.0[2], det.0[3], det.1, det.2
        );
    }

    Ok(())
}

// Helper functions

fn create_dummy_image() -> Tensor<RustGpu, 4, f32> {
    // Create random-like input data
    Tensor::create([1, 3, 416, 416])
}

fn leaky_relu(input: &Tensor<RustGpu, 4, f32>, alpha: f32) -> AbResult<Tensor<RustGpu, 4, f32>> {
    input.leaky_relu(alpha)
}

fn residual_block(
    input: &Tensor<RustGpu, 4, f32>,
    channels: usize,
) -> AbResult<Tensor<RustGpu, 4, f32>> {
    // Residual block: 1x1 conv -> 3x3 conv -> add residual
    let conv1_weights: Tensor<RustGpu, 4, f32> = Tensor::create([channels / 2, channels, 1, 1]);
    let conv1_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([channels / 2]);
    let conv1_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([channels / 2]);

    let x = input.conv_2d(
        &conv1_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([0, 0]),
        },
    )?;
    let x = x.batch_norm(&conv1_bn_gamma, &conv1_bn_beta, BatchNormParams::default())?;
    let x = leaky_relu(&x, 0.1)?;

    let conv2_weights: Tensor<RustGpu, 4, f32> = Tensor::create([channels, channels / 2, 3, 3]);
    let conv2_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([channels]);
    let conv2_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([channels]);

    let x = x.conv_2d(
        &conv2_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]),
        },
    )?;
    let x = x.batch_norm(&conv2_bn_gamma, &conv2_bn_beta, BatchNormParams::default())?;
    let x = leaky_relu(&x, 0.1)?;

    // Add residual connection
    let result = x.add(input)?;
    Ok(result)
}

fn detection_head(
    input: &Tensor<RustGpu, 4, f32>,
    in_channels: usize,
    out_channels: usize,
) -> AbResult<Tensor<RustGpu, 4, f32>> {
    // Detection head: several conv layers ending with final prediction layer

    // Conv layer 1
    let conv1_weights: Tensor<RustGpu, 4, f32> =
        Tensor::create([in_channels / 2, in_channels, 1, 1]);
    let conv1_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([in_channels / 2]);
    let conv1_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([in_channels / 2]);

    let x = input.conv_2d(
        &conv1_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([0, 0]),
        },
    )?;
    let x = x.batch_norm(&conv1_bn_gamma, &conv1_bn_beta, BatchNormParams::default())?;
    let x = leaky_relu(&x, 0.1)?;

    // Conv layer 2
    let conv2_weights: Tensor<RustGpu, 4, f32> =
        Tensor::create([in_channels, in_channels / 2, 3, 3]);
    let conv2_bn_gamma: Tensor<RustGpu, 1, f32> = Tensor::create([in_channels]);
    let conv2_bn_beta: Tensor<RustGpu, 1, f32> = Tensor::create([in_channels]);

    let x = x.conv_2d(
        &conv2_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]),
        },
    )?;
    let x = x.batch_norm(&conv2_bn_gamma, &conv2_bn_beta, BatchNormParams::default())?;
    let x = leaky_relu(&x, 0.1)?;

    // Final prediction layer (no batch norm, no activation)
    let final_weights: Tensor<RustGpu, 4, f32> = Tensor::create([out_channels, in_channels, 1, 1]);
    let output = x.conv_2d(
        &final_weights,
        Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([0, 0]),
        },
    )?;

    Ok(output)
}

fn upsample(input: &Tensor<RustGpu, 4, f32>, scale: usize) -> AbResult<Tensor<RustGpu, 4, f32>> {
    input.upsample(scale)
}

fn concatenate(
    a: &Tensor<RustGpu, 4, f32>,
    b: &Tensor<RustGpu, 4, f32>,
    dim: usize,
) -> AbResult<Tensor<RustGpu, 4, f32>> {
    a.concatenate(b, dim)
}

fn element_wise_add(
    a: &Tensor<RustGpu, 4, f32>,
    b: &Tensor<RustGpu, 4, f32>,
) -> AbResult<Tensor<RustGpu, 4, f32>> {
    a.add(b)
}

// Detection struct: (bbox, confidence, class_id)
type Detection = ([f32; 4], f32, usize);

fn decode_yolo_outputs(predictions: &[Tensor<RustGpu, 4, f32>]) -> AbResult<Vec<Detection>> {
    // Decode YOLO predictions into bounding boxes
    let mut detections = Vec::new();

    // COCO anchors for each scale
    let anchors = vec![
        vec![10.0, 13.0, 16.0, 30.0, 33.0, 23.0],      // 13x13
        vec![30.0, 61.0, 62.0, 45.0, 59.0, 119.0],     // 26x26
        vec![116.0, 90.0, 156.0, 198.0, 373.0, 326.0], // 52x52
    ];

    let strides = vec![32, 16, 8]; // Downsampling ratios
    let num_classes = 80; // COCO classes

    for (scale_idx, pred) in predictions.iter().enumerate() {
        let pred_data = pred.to_vec();
        let shape = pred.shape().as_slice();
        let grid_h = shape[2];
        let grid_w = shape[3];

        // Decode predictions for this scale
        for h in 0..grid_h {
            for w in 0..grid_w {
                for anchor_idx in 0..3 {
                    let stride = strides[scale_idx];
                    let anchor_w = anchors[scale_idx][anchor_idx * 2];
                    let anchor_h = anchors[scale_idx][anchor_idx * 2 + 1];

                    // Calculate index in flattened array
                    let base_idx = (anchor_idx * (5 + num_classes) * grid_h + h) * grid_w + w;

                    // Extract box parameters (sigmoid for x,y and confidence)
                    let x = sigmoid(pred_data[base_idx]);
                    let y = sigmoid(pred_data[base_idx + 1]);
                    let w_pred = pred_data[base_idx + 2].exp();
                    let h_pred = pred_data[base_idx + 3].exp();
                    let confidence = sigmoid(pred_data[base_idx + 4]);

                    if confidence > 0.5 {
                        // Convert to absolute coordinates
                        let center_x = (w as f32 + x) * stride as f32;
                        let center_y = (h as f32 + y) * stride as f32;
                        let width = w_pred * anchor_w;
                        let height = h_pred * anchor_h;

                        // Find best class
                        let mut best_class = 0;
                        let mut best_score = 0.0;

                        for class_idx in 0..num_classes {
                            let class_score = sigmoid(pred_data[base_idx + 5 + class_idx]);
                            if class_score > best_score {
                                best_score = class_score;
                                best_class = class_idx;
                            }
                        }

                        let final_score = confidence * best_score;
                        if final_score > 0.3 {
                            detections.push((
                                [
                                    center_x - width / 2.0,
                                    center_y - height / 2.0,
                                    width,
                                    height,
                                ],
                                final_score,
                                best_class,
                            ));
                        }
                    }
                }
            }
        }
    }

    Ok(detections)
}

fn apply_nms(detections: &[Detection], iou_threshold: f32) -> AbResult<Vec<Detection>> {
    // Non-Maximum Suppression
    let mut sorted_detections = detections.to_vec();
    sorted_detections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // Sort by confidence

    let mut keep = Vec::new();
    let mut suppressed = vec![false; sorted_detections.len()];

    for i in 0..sorted_detections.len() {
        if suppressed[i] {
            continue;
        }

        keep.push(sorted_detections[i].clone());

        // Suppress overlapping detections of the same class
        for j in (i + 1)..sorted_detections.len() {
            if suppressed[j] || sorted_detections[i].2 != sorted_detections[j].2 {
                continue;
            }

            let iou = calculate_iou(&sorted_detections[i].0, &sorted_detections[j].0);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Ok(keep)
}

fn calculate_iou(box1: &[f32; 4], box2: &[f32; 4]) -> f32 {
    let x1_min = box1[0];
    let y1_min = box1[1];
    let x1_max = box1[0] + box1[2];
    let y1_max = box1[1] + box1[3];

    let x2_min = box2[0];
    let y2_min = box2[1];
    let x2_max = box2[0] + box2[2];
    let y2_max = box2[1] + box2[3];

    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    if inter_x_max <= inter_x_min || inter_y_max <= inter_y_min {
        return 0.0;
    }

    let inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    let box1_area = box1[2] * box1[3];
    let box2_area = box2[2] * box2[3];
    let union_area = box1_area + box2_area - inter_area;

    inter_area / union_area
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
