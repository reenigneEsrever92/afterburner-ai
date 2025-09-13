//! GPU-Accelerated Edge Detection Example
//!
//! This example demonstrates edge detection using the Afterburner AI framework
//! with GPU-accelerated conversions and CPU-based convolution as fallback.
//!
//! Features:
//! - Image loading and display using egui
//! - GPU-based u8 to f32 conversion
//! - Sobel edge detection using Conv2D operations
//! - GPU-based f32 to u8 conversion for display
//! - Side-by-side image comparison
//! - Real-time processing with progress indication
//!
//! The implementation uses:
//! - GPU convert operations for data type conversion
//! - 4D tensors with shape [batch=1, channels=1, height, width]
//! - Sobel X kernel for horizontal edge detection
//! - Conv2D operations with stride [1, 1] and padding [1, 1]
//! - Mixed GPU/CPU pipeline for reliable processing

use afterburner_cpu::prelude::*;
use afterburner_rustgpu::prelude::*;
use eframe::egui as ef;
use egui::{ColorImage, TextureOptions};
use image::{EncodableLayout, ImageBuffer, Rgb};

fn main() {
    afterburner_rustgpu::init();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "GPU-Enhanced Edge Detection",
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

    fn apply_gpu_edge_detection(&mut self) {
        if self.processing {
            return;
        }

        self.processing = true;
        self.processing_method = Some("GPU Convert + CPU Conv2D".to_string());

        let width = self.original_img.width() as usize;
        let height = self.original_img.height() as usize;

        // Step 1: Convert RGB image to grayscale using CPU (reliable)
        let mut grayscale_u8: Vec<u8> = Vec::with_capacity(width * height);
        for pixel in self.original_img.pixels() {
            // Convert RGB to grayscale using luminance formula
            let gray =
                (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            grayscale_u8.push(gray);
        }

        // Step 2: Use GPU to convert u8 to f32
        let grayscale_u8_tensor: Tensor<RustGpu, 1, u8> =
            RustGpu::new_tensor(Shape([width * height]), grayscale_u8);

        let grayscale_f32_tensor: Tensor<RustGpu, 1, f32> = grayscale_u8_tensor.convert();

        // Step 3: Copy to CPU for convolution (since GPU conv has issues)
        let grayscale_data = grayscale_f32_tensor.to_vec();
        let input_tensor: Tensor<afterburner_cpu::Cpu, 4, f32> = afterburner_cpu::Cpu::new_tensor(
            Shape([1, 1, height, width]),
            grayscale_data.iter().map(|&x| x / 255.0).collect(), // Normalize to 0-1
        );

        // Step 4: Create Sobel kernel on CPU
        let sobel_x_data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let kernel_x: Tensor<afterburner_cpu::Cpu, 4, f32> =
            afterburner_cpu::Cpu::new_tensor(Shape([1, 1, 3, 3]), sobel_x_data);

        // Step 5: Apply CPU convolution with padding
        let conv_params = Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]), // Maintain input size
        };

        let edges_result = input_tensor.conv_2d(&kernel_x, conv_params);

        match edges_result {
            Ok(edges_tensor) => {
                let edge_data = edges_tensor.to_vec();

                // Step 6: Find the data range for proper scaling
                let min_val = edge_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_val = edge_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Step 7: Scale edge values and convert to f32 tensor on GPU
                let mut scaled_edges = Vec::with_capacity(edge_data.len());
                for &edge_val in &edge_data {
                    // Scale to 0-255 range
                    let scaled = if max_val > min_val {
                        ((edge_val.abs() - min_val.abs()) / (max_val.abs() - min_val.abs()) * 255.0)
                            .clamp(0.0, 255.0)
                    } else {
                        (edge_val.abs() * 255.0).clamp(0.0, 255.0)
                    };
                    scaled_edges.push(scaled);
                }

                let edge_f32_tensor: Tensor<RustGpu, 1, f32> =
                    RustGpu::new_tensor(Shape([width * height]), scaled_edges);

                // Step 8: Use GPU to convert f32 to u8
                let edge_u8_tensor: Tensor<RustGpu, 1, u8> = edge_f32_tensor.convert();

                // Step 9: Get final data and create RGB image
                let edge_u8_data = edge_u8_tensor.to_vec();
                let mut rgb_data = Vec::with_capacity(width * height * 3);

                for &intensity in &edge_u8_data {
                    rgb_data.push(intensity); // R
                    rgb_data.push(intensity); // G
                    rgb_data.push(intensity); // B
                }

                // Step 10: Create edge detected image
                self.edge_img = ImageBuffer::from_raw(width as u32, height as u32, rgb_data);
            }
            Err(e) => {
                eprintln!("Edge detection failed: {:?}", e);
                self.edge_img = None;
            }
        }

        self.processing = false;
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &ef::Context, _frame: &mut eframe::Frame) {
        ef::CentralPanel::default().show(ctx, |ui| {
            ui.heading("GPU-Enhanced Edge Detection");

            ui.horizontal(|ui| {
                if ui.button("Apply GPU-Enhanced Edge Detection").clicked() && !self.processing {
                    self.apply_gpu_edge_detection();
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
                    if !self.processing && self.edge_img.is_some() {
                        ui.label(format!("✓ Completed using {}", method));
                    }
                }
            });

            ui.separator();

            ui.horizontal(|ui| {
                // Original image
                ui.vertical(|ui| {
                    ui.heading("Original Image");
                    let original_bytes = self.original_img.as_bytes();
                    let original_size = [
                        self.original_img.width() as usize,
                        self.original_img.height() as usize,
                    ];

                    let original_image = ColorImage::from_rgb(original_size, original_bytes);
                    let original_texture =
                        ctx.load_texture("original", original_image, TextureOptions::default());
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
                        ui.heading("GPU-Enhanced Edge Detection Result");
                        let edge_bytes = edge_img.as_bytes();
                        let edge_size = [edge_img.width() as usize, edge_img.height() as usize];

                        // Verify the data length matches expected RGB format
                        let expected_len = edge_size[0] * edge_size[1] * 3;
                        if edge_bytes.len() == expected_len {
                            let edge_image = ColorImage::from_rgb(edge_size, edge_bytes);
                            let edge_texture =
                                ctx.load_texture("edges", edge_image, TextureOptions::default());
                            ui.add(
                                egui::Image::new(&edge_texture)
                                    .corner_radius(5)
                                    .max_height(400.0)
                                    .max_width(400.0),
                            );
                        } else {
                            ui.label(format!(
                                "Error: Image data size mismatch. Expected {}, got {}",
                                expected_len,
                                edge_bytes.len()
                            ));
                        }
                    });
                } else {
                    ui.vertical(|ui| {
                        ui.heading("GPU-Enhanced Edge Detection Result");
                        ui.label("Click 'Apply GPU-Enhanced Edge Detection' to process the image");
                    });
                }
            });

            if let Some(ref method) = self.processing_method {
                if self.edge_img.is_some() {
                    ui.label(format!(
                        "✓ Pipeline completed successfully using: {}",
                        method
                    ));
                } else if !self.processing {
                    ui.label("❌ Pipeline failed. Check console for error details.");
                }
            }
        });
    }
}
