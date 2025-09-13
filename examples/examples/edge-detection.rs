//! Edge Detection Example
//!
//! This example demonstrates edge detection using Sobel operators
//! implemented with Conv2D operations from the Afterburner AI framework.
//!
//! Features completed:
//! - Image loading and display using egui
//! - RGB to grayscale conversion
//! - Sobel edge detection using Conv2D with CPU backend
//! - Batch normalization applied to edge detection results
//! - Side-by-side image comparison
//! - Real-time processing with progress indication
//! - Padding support to maintain input image dimensions
//! - Adaptive scaling for optimal edge visibility
//!
//! The implementation uses:
//! - 4D tensors with shape [batch=1, channels=1, height, width]
//! - Sobel X kernel for horizontal edge detection: [-1, 0, 1, -2, 0, 2, -1, 0, 1]
//! - Conv2D operations with stride [1, 1] and padding [1, 1]
//! - Batch normalization for stable edge detection results
//! - CPU backend for reliable processing of large images
//! - Automatic scaling to convert edge values to visible grayscale intensities

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
        self.processing_method = Some("CPU Conv2D + BatchNorm".to_string());

        // Convert RGB image to grayscale for edge detection
        let width = self.original_img.width() as usize;
        let height = self.original_img.height() as usize;

        let mut grayscale: Vec<f32> = Vec::with_capacity(width * height);
        for pixel in self.original_img.pixels() {
            // Convert RGB to grayscale using luminance formula
            let gray = 0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32;
            grayscale.push(gray / 255.0); // Normalize to 0-1 range
        }

        // Create CPU tensors for reliable processing
        let input_tensor: Tensor<afterburner_cpu::Cpu, 4, f32> =
            afterburner_cpu::Cpu::new_tensor(Shape([1, 1, height, width]), grayscale);

        // Sobel X kernel for horizontal edge detection
        let sobel_x_data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let kernel_x: Tensor<afterburner_cpu::Cpu, 4, f32> =
            afterburner_cpu::Cpu::new_tensor(Shape([1, 1, 3, 3]), sobel_x_data);

        // Apply convolution with Sobel kernels
        let conv_params = Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]), // Add padding to maintain input size
        };

        // Apply edge detection using CPU backend
        let edges_result = input_tensor.conv_2d(&kernel_x, conv_params);

        // Apply batch normalization to the edge detection results
        let bn_edges_result = edges_result.and_then(|edges_tensor| {
            // Create gamma and beta parameters for batch normalization
            // Single channel output from Sobel, so we need 1-dimensional gamma and beta
            let gamma: Tensor<afterburner_cpu::Cpu, 1, f32> =
                afterburner_cpu::Cpu::new_tensor(Shape([1]), vec![1.0]); // No scaling
            let beta: Tensor<afterburner_cpu::Cpu, 1, f32> =
                afterburner_cpu::Cpu::new_tensor(Shape([1]), vec![0.0]); // No shift

            // Apply batch normalization with default parameters (epsilon = 1e-5)
            edges_tensor.batch_norm(&gamma, &beta, BatchNormParams::default())
        });

        match bn_edges_result {
            Ok(edges_tensor) => {
                // Get the edge data from the tensor
                let edge_data = edges_tensor.to_vec();
                let edge_height = height; // With padding, output size equals input size
                let edge_width = width;

                // Ensure we have the expected amount of data
                if edge_data.len() == edge_width * edge_height {
                    // Convert edge data back to RGB image
                    let mut edge_pixels: Vec<u8> = Vec::with_capacity(edge_width * edge_height * 3);

                    // Find the data range for proper scaling
                    let min_val = edge_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let max_val = edge_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                    // Use adaptive scaling to make edges visible
                    let scale_factor = if max_val > min_val && max_val.abs() > 0.001 {
                        255.0 / max_val.abs().max(min_val.abs())
                    } else {
                        255.0
                    };

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
                        "Convolution returned unexpected data size: expected {}, got {}",
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
            ui.label("The implementation uses CPU-based convolution with padding and batch normalization for reliable edge detection.");

            if let Some(ref method) = self.processing_method {
                if self.edge_img.is_some() {
                    ui.label(format!("Last processing method: {}", method));
                } else {
                    ui.label("Edge detection failed. Please try again or check the console for error messages.");
                }
            }
        });
    }
}
