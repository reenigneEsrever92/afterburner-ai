//! Fully GPU-Accelerated Edge Detection Example
//!
//! This example demonstrates a complete GPU-based edge detection pipeline using
//! the Afterburner AI framework with RustGPU compute shaders.
//!
//! Features:
//! - Pure GPU pipeline - all operations run on GPU compute shaders
//! - GPU-based RGB to grayscale conversion
//! - GPU-accelerated Sobel edge detection using Conv2D
//! - GPU-based batch normalization for stable results
//! - GPU-based grayscale to RGB conversion for display
//! - Real-time processing with progress indication
//! - Side-by-side image comparison
//!
//! GPU Pipeline:
//! 1. Load RGB image data to GPU
//! 2. GPU RGB → Grayscale conversion shader
//! 3. GPU Sobel Conv2D with padding
//! 4. GPU Batch normalization
//! 5. GPU Grayscale → RGB conversion shader
//! 6. Display results

use afterburner_rustgpu::prelude::*;
use eframe::egui as ef;
use egui::{ColorImage, TextureOptions};
use image::{EncodableLayout, ImageBuffer, Rgb};

fn main() {
    afterburner_rustgpu::init();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Pure GPU Edge Detection",
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
        self.processing_method = Some("Pure GPU Pipeline".to_string());

        let width = self.original_img.width() as usize;
        let height = self.original_img.height() as usize;

        // Step 1: Create RGB tensor on GPU from image data
        let rgb_data: Vec<u8> = self.original_img.as_raw().to_vec();
        let rgb_tensor: Tensor<RustGpu, 1, u8> =
            RustGpu::new_tensor(Shape([width * height * 3]), rgb_data);

        // Step 2: GPU RGB to Grayscale conversion
        let grayscale_tensor: Tensor<RustGpu, 1, f32> =
            RustGpu::rgb_to_grayscale(&rgb_tensor, width, height);

        // Step 3: Reshape to 4D tensor for convolution [batch=1, channels=1, height, width]
        let grayscale_data = grayscale_tensor.to_vec();
        let input_4d: Tensor<RustGpu, 4, f32> =
            RustGpu::new_tensor(Shape([1, 1, height, width]), grayscale_data);

        // Step 4: Create Sobel X kernel on GPU
        let sobel_x_data = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let kernel_x: Tensor<RustGpu, 4, f32> =
            RustGpu::new_tensor(Shape([1, 1, 3, 3]), sobel_x_data);

        // Step 5: GPU Conv2D with padding to maintain image size
        let conv_params = Conv2DParams {
            stride: Shape([1, 1]),
            padding: Shape([1, 1]),
        };

        let conv_result = input_4d.conv_2d(&kernel_x, conv_params);

        match conv_result {
            Ok(edges_tensor) => {
                // Step 6: GPU Batch Normalization for stable edge values
                let gamma: Tensor<RustGpu, 1, f32> = RustGpu::new_tensor(Shape([1]), vec![3.0]); // Scale for visibility
                let beta: Tensor<RustGpu, 1, f32> = RustGpu::new_tensor(Shape([1]), vec![0.0]); // No shift

                let bn_result = edges_tensor.batch_norm(&gamma, &beta, BatchNormParams::default());

                match bn_result {
                    Ok(normalized_edges) => {
                        // Step 7: Reshape to 1D for image conversion
                        let edge_data = normalized_edges.to_vec();
                        let edge_1d: Tensor<RustGpu, 1, f32> =
                            RustGpu::new_tensor(Shape([width * height]), edge_data);

                        // Step 8: GPU Grayscale to RGB conversion
                        let rgb_edges: Tensor<RustGpu, 1, u8> =
                            RustGpu::grayscale_to_rgb(&edge_1d, width, height);

                        // Step 9: Get final RGB data from GPU
                        let final_rgb_data = rgb_edges.to_vec();

                        // Step 10: Create final edge-detected image
                        self.edge_img =
                            ImageBuffer::from_raw(width as u32, height as u32, final_rgb_data);

                        if self.edge_img.is_none() {
                            eprintln!("Failed to create final image from GPU data");
                        }
                    }
                    Err(e) => {
                        eprintln!("GPU Batch Normalization failed: {:?}", e);
                        self.fallback_cpu_processing(width, height);
                    }
                }
            }
            Err(e) => {
                eprintln!("GPU Conv2D failed: {:?}", e);
                self.fallback_cpu_processing(width, height);
            }
        }

        self.processing = false;
    }

    fn fallback_cpu_processing(&mut self, width: usize, height: usize) {
        eprintln!("Falling back to simple CPU edge detection...");
        self.processing_method = Some("CPU Fallback".to_string());

        let mut edge_pixels: Vec<u8> = Vec::with_capacity(width * height * 3);

        for y in 0..height {
            for x in 0..width {
                let current = self.original_img.get_pixel(x as u32, y as u32);
                let gray = (0.299 * current[0] as f32
                    + 0.587 * current[1] as f32
                    + 0.114 * current[2] as f32) as u8;

                let mut edge_strength = 0u8;

                // Simple edge detection using pixel differences
                if x > 0 {
                    let left = self.original_img.get_pixel((x - 1) as u32, y as u32);
                    let left_gray = (0.299 * left[0] as f32
                        + 0.587 * left[1] as f32
                        + 0.114 * left[2] as f32) as u8;
                    edge_strength = edge_strength.saturating_add(gray.abs_diff(left_gray));
                }

                if y > 0 {
                    let top = self.original_img.get_pixel(x as u32, (y - 1) as u32);
                    let top_gray = (0.299 * top[0] as f32
                        + 0.587 * top[1] as f32
                        + 0.114 * top[2] as f32) as u8;
                    edge_strength = edge_strength.saturating_add(gray.abs_diff(top_gray));
                }

                let intensity = (edge_strength as f32 * 2.0).min(255.0) as u8;
                edge_pixels.push(intensity); // R
                edge_pixels.push(intensity); // G
                edge_pixels.push(intensity); // B
            }
        }

        self.edge_img = ImageBuffer::from_raw(width as u32, height as u32, edge_pixels);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &ef::Context, _frame: &mut eframe::Frame) {
        ef::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🚀 Pure GPU Edge Detection");

            ui.horizontal(|ui| {
                if ui.button("🔥 Apply GPU Edge Detection").clicked() && !self.processing {
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
                        ui.label(format!("✅ Completed using {}", method));
                    }
                }
            });

            ui.separator();

            ui.horizontal(|ui| {
                // Original image
                ui.vertical(|ui| {
                    ui.heading("📸 Original Image");
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
                            .corner_radius(8.0)
                            .max_height(400.0)
                            .max_width(400.0),
                    );
                });

                // Edge detected image
                if let Some(ref edge_img) = self.edge_img {
                    ui.vertical(|ui| {
                        ui.heading("⚡ GPU Edge Detection Result");
                        let edge_bytes = edge_img.as_bytes();
                        let edge_size = [edge_img.width() as usize, edge_img.height() as usize];

                        let expected_len = edge_size[0] * edge_size[1] * 3;
                        if edge_bytes.len() == expected_len {
                            let edge_image = ColorImage::from_rgb(edge_size, edge_bytes);
                            let edge_texture =
                                ctx.load_texture("edges", edge_image, TextureOptions::default());

                            ui.add(
                                egui::Image::new(&edge_texture)
                                    .corner_radius(8.0)
                                    .max_height(400.0)
                                    .max_width(400.0),
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::RED,
                                format!(
                                    "⚠️ Image data error: Expected {} bytes, got {}",
                                    expected_len,
                                    edge_bytes.len()
                                ),
                            );
                        }
                    });
                } else {
                    ui.vertical(|ui| {
                        ui.heading("⚡ GPU Edge Detection Result");
                        ui.label("👆 Click 'Apply GPU Edge Detection' to see the magic!");
                    });
                }
            });

            ui.separator();

            ui.label("🎯 Pure GPU Pipeline Architecture:");
            ui.label("1. 📤 Load RGB image data to GPU memory");
            ui.label("2. 🔄 GPU RGB → Grayscale conversion (compute shader)");
            ui.label("3. 🎯 GPU Sobel edge detection with Conv2D + padding");
            ui.label("4. ⚖️ GPU Batch normalization for stable results");
            ui.label("5. 🎨 GPU Grayscale → RGB conversion (compute shader)");
            ui.label("6. 📥 Transfer final results back for display");

            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(0, 150, 255),
                "🚀 All tensor operations execute on GPU compute shaders for maximum performance!",
            );

            if let Some(ref method) = self.processing_method {
                if self.edge_img.is_some() && !self.processing {
                    match method.as_str() {
                        "Pure GPU Pipeline" => {
                            ui.colored_label(
                                egui::Color32::GREEN,
                                "🔥 GPU pipeline executed successfully!",
                            );
                        }
                        "CPU Fallback" => {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "⚠️ GPU pipeline failed, used CPU fallback",
                            );
                        }
                        _ => {
                            ui.label(format!("Processing completed with: {}", method));
                        }
                    }
                }
            }
        });
    }
}
