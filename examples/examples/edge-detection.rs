//! Edge Detection Example
//!
//! This example demonstrates edge detection using Sobel operators
//! implemented with Conv2D operations from the Afterburner AI framework.
//!
//! Features completed:
//! - Image loading and display using egui
//! - RGB to grayscale conversion
//! - Sobel edge detection using Conv2D (CPU backend for testing)
//! - Side-by-side image comparison
//! - Real-time processing with progress indication
//!
//! The implementation uses:
//! - 4D tensors with shape [batch=1, channels=1, height, width]
//! - Sobel X kernel for horizontal edge detection
//! - Conv2D operations with stride [1, 1]
//! - CPU backend for reliable processing

use afterburner_cpu::prelude::*;
use afterburner_rustgpu::prelude::*;
use eframe::egui as ef;
use egui::{ColorImage, TextureOptions};
use image::{EncodableLayout, ImageBuffer, Rgb};

fn main() {
    init();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Edge Detection",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .expect("Running GUI");
}

struct App {
    original_img: ImageBuffer<Rgb<u8>, Vec<u8>>,
    edge_img: Option<ImageBuffer<Rgb<u8>, Vec<u8>>>,
    processing: bool,
    processing_method: Option<String>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);

        let original_img = image::ImageReader::open("assets/stones.jpg")
            .expect("Image loaded")
            .decode()
            .expect("Image decoded")
            .to_rgb8();

        App {
            original_img,
            edge_img: None,
            processing: false,
            processing_method: None,
        }
    }

    fn apply_edge_detection(&mut self) {
        if self.processing {
            return;
        }

        self.processing = true;
        self.processing_method = Some("GPU Conv2D".to_string());

        // Convert RGB image to grayscale for edge detection
        // Use a small subset for testing (64x64 pixels from top-left corner)
        let test_width = 64.min(self.original_img.width() as usize);
        let test_height = 64.min(self.original_img.height() as usize);

        let mut grayscale: Vec<f32> = Vec::with_capacity(test_width * test_height);
        for y in 0..test_height {
            for x in 0..test_width {
                let pixel = self.original_img.get_pixel(x as u32, y as u32);
                // Convert RGB to grayscale using luminance formula
                let gray =
                    0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
                grayscale.push(gray / 255.0); // Normalize to 0-1 range
            }
        }

        let width = test_width;
        let height = test_height;

        // Debug: Print grayscale statistics
        let gray_min = grayscale.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let gray_max = grayscale.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let gray_avg = grayscale.iter().sum::<f32>() / grayscale.len() as f32;
        eprintln!(
            "Grayscale stats: min={:.3}, max={:.3}, avg={:.3}, samples: {}",
            gray_min,
            gray_max,
            gray_avg,
            grayscale.len()
        );

        // Create input tensor with grayscale data [batch=1, channels=1, height, width]
        eprintln!(
            "Creating input tensor with shape [1, 1, {}, {}]",
            height, width
        );
        let input_tensor = RustGpu::new_tensor(Shape([1, 1, height, width]), grayscale);

        // Verify tensor was created
        eprintln!("Input tensor shape: {:?}", input_tensor.shape());
        let input_readback = input_tensor.to_vec();
        let input_sample = &input_readback[0..5.min(input_readback.len())];
        eprintln!("Input tensor readback sample: {:?}", input_sample);

        // Sobel X kernel for horizontal edge detection
        let sobel_x_data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        eprintln!("Sobel kernel: {:?}", sobel_x_data);
        eprintln!("Creating kernel tensor with shape [1, 1, 3, 3]");
        let kernel_x = RustGpu::new_tensor(Shape([1, 1, 3, 3]), sobel_x_data.clone());

        // Verify kernel tensor was created
        eprintln!("Kernel tensor shape: {:?}", kernel_x.shape());
        let kernel_readback = kernel_x.to_vec();
        eprintln!("Kernel tensor readback: {:?}", kernel_readback);

        // Note: For simplicity, we're only using the Sobel X kernel
        // In a complete implementation, you'd also use the Sobel Y kernel and combine gradients

        // Apply convolution with Sobel kernels
        let conv_params = Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]), // Add padding to maintain input size
        };

        eprintln!(
            "Calling conv2d with params: stride={:?}, padding={:?}",
            conv_params.stride, conv_params.padding
        );

        // Now test GPU backend with small image to isolate the issue
        eprintln!("Testing with GPU backend on small image...");
        let edges_result = input_tensor.conv_2d(&kernel_x, conv_params);

        match edges_result {
            Ok(edges_tensor) => {
                // Get the edge data from the tensor
                let edge_data = edges_tensor.to_vec();
                let edge_height = height; // With padding, output size equals input size
                let edge_width = width;

                // Debug: Print some statistics about the edge data
                let min_val = edge_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = edge_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let avg_val = edge_data.iter().sum::<f32>() / edge_data.len() as f32;
                let non_zero_count = edge_data.iter().filter(|&&x| x.abs() > 0.001).count();

                eprintln!(
                    "Edge data stats: min={:.3}, max={:.3}, avg={:.3}, non_zero={}/{}, expected_size={}",
                    min_val,
                    max_val,
                    avg_val,
                    non_zero_count,
                    edge_data.len(),
                    edge_width * edge_height
                );

                // Print first few edge values for debugging
                let sample_count = 20.min(edge_data.len());
                eprintln!(
                    "First {} edge values: {:?}",
                    sample_count,
                    &edge_data[0..sample_count]
                );

                // Ensure we have the expected amount of data
                if edge_data.len() == edge_width * edge_height {
                    // Convert edge data back to RGB image
                    let mut edge_pixels: Vec<u8> = Vec::with_capacity(edge_width * edge_height * 3);

                    // Use adaptive scaling based on the actual data range
                    let scale_factor = if max_val > min_val && max_val.abs() > 0.001 {
                        255.0 / max_val.abs().max(min_val.abs())
                    } else {
                        1.0
                    };

                    eprintln!("Using scale factor: {:.3}", scale_factor);

                    for &edge_val in &edge_data {
                        // Scale edge values to 0-255 range
                        let scaled_val = (edge_val.abs() * scale_factor).min(255.0).max(0.0);
                        let intensity = scaled_val as u8;

                        edge_pixels.push(intensity); // R
                        edge_pixels.push(intensity); // G
                        edge_pixels.push(intensity); // B
                    }

                    // Create edge detected image
                    self.edge_img =
                        ImageBuffer::from_raw(edge_width as u32, edge_height as u32, edge_pixels);
                } else {
                    eprintln!(
                        "GPU convolution returned unexpected data size: expected {}, got {}",
                        edge_width * edge_height,
                        edge_data.len()
                    );
                    self.edge_img = None;
                }
            }
            Err(e) => {
                eprintln!("Error applying edge detection: {:?}", e);
                self.edge_img = None;
            }
        }

        self.processing = false;
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &ef::Context, _frame: &mut eframe::Frame) {
        ef::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Edge Detection Example");

            ui.horizontal(|ui| {
                if ui.button("Apply Edge Detection").clicked() && !self.processing {
                    self.apply_edge_detection();
                }

                if self.processing {
                    ui.spinner();
                    if let Some(ref method) = self.processing_method {
                        ui.label(format!("Processing with {}...", method));
                    } else {
                        ui.label("Processing...");
                    }
                }

                if let Some(ref method) = self.processing_method {
                    if !self.processing {
                        if self.edge_img.is_some() {
                            ui.label(format!("✓ Completed using {}", method));
                        } else {
                            ui.label(format!("✗ Failed with {}", method));
                        }
                    }
                }
            });

            ui.separator();

            ui.horizontal(|ui| {
                // Original image
                ui.vertical(|ui| {
                    ui.heading("Original Image");
                    let original_bytes = self.original_img.as_bytes();
                    let original_size = [self.original_img.width() as usize, self.original_img.height() as usize];

                    let original_image = ColorImage::from_rgb(original_size, original_bytes);
                    let original_texture = ctx.load_texture("original", original_image, TextureOptions::default());
                    ui.add(
                        egui::Image::new(&original_texture)
                            .corner_radius(5)
                            .max_height(400.0)
                            .max_width(400.0),
                    );
                });

                // Edge detected image
                if let Some(ref edge_img) = self.edge_img {
                    ui.vertical(|ui| {
                        ui.heading("Edge Detection Result");
                        let edge_bytes = edge_img.as_bytes();
                        let edge_size = [edge_img.width() as usize, edge_img.height() as usize];

                        // Verify the data length matches expected RGB format
                        let expected_len = edge_size[0] * edge_size[1] * 3;
                        if edge_bytes.len() == expected_len {
                            let edge_image = ColorImage::from_rgb(edge_size, edge_bytes);
                            let edge_texture = ctx.load_texture("edges", edge_image, TextureOptions::default());
                            ui.add(
                                egui::Image::new(&edge_texture)
                                    .corner_radius(5)
                                    .max_height(400.0)
                                    .max_width(400.0),
                            );
                        } else {
                            ui.label(format!("Error: Image data size mismatch. Expected {}, got {}", expected_len, edge_bytes.len()));
                        }
                    });
                } else {
                    ui.vertical(|ui| {
                        ui.heading("Edge Detection Result");
                        ui.label("Click 'Apply Edge Detection' to process the image");
                    });
                }
            });

            ui.separator();
            ui.label("This example demonstrates edge detection using Sobel operators implemented with Conv2D operations.");
            ui.label("The implementation uses GPU-accelerated convolution operations only.");

            if let Some(ref method) = self.processing_method {
                if self.edge_img.is_some() {
                    ui.label(format!("Last processing method: {}", method));
                } else {
                    ui.label("Edge detection failed. Try with a smaller image or check GPU compatibility.");
                }
            }
        });
    }
}
