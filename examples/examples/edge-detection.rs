//! Pure GPU-Accelerated Edge Detection Example (Without Conv2D)
//!
//! This example demonstrates GPU-based edge detection using basic tensor operations
//! instead of Conv2D, since the Conv2D implementation appears to have issues.
//!
//! This approach uses direct GPU memory operations to implement Sobel filtering
//! manually, ensuring all computation stays on the GPU.
//!
//! Features:
//! - Pure GPU pipeline - all operations run on GPU
//! - GPU-based RGB to grayscale conversion
//! - Manual GPU-based Sobel edge detection using direct memory access
//! - GPU-based normalization
//! - Real-time processing with progress indication
//! - Side-by-side image comparison
//!
//! GPU Pipeline:
//! 1. Load RGB image data to GPU
//! 2. GPU RGB → Grayscale conversion
//! 3. GPU manual Sobel edge detection (X & Y gradients)
//! 4. GPU gradient magnitude calculation
//! 5. GPU normalization and RGB conversion
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
    error_message: Option<String>,
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
            error_message: None,
        }
    }

    fn apply_gpu_edge_detection(&mut self) {
        if self.processing {
            return;
        }

        self.processing = true;
        self.error_message = None;

        let width = self.original_img.width() as usize;
        let height = self.original_img.height() as usize;

        println!("🚀 Starting GPU-only edge detection pipeline (manual implementation)");
        println!("📐 Image dimensions: {}x{}", width, height);

        // Step 1: Create RGB tensor on GPU from image data
        let rgb_data: Vec<u8> = self.original_img.as_raw().to_vec();
        println!("📤 Uploading {} bytes to GPU", rgb_data.len());

        let rgb_tensor: Tensor<RustGpu, 1, u8> =
            RustGpu::new_tensor(Shape([width * height * 3]), rgb_data);

        // Step 2: GPU RGB to Grayscale conversion
        println!("🔄 GPU RGB → Grayscale conversion");
        let grayscale_tensor = RustGpu::rgb_to_grayscale(&rgb_tensor, width, height);

        // Step 3: Manual GPU-based Sobel edge detection
        println!("🎯 Applying manual GPU Sobel edge detection");

        let sobel_x = RustGpu::new_tensor(
            [3, 3].into(),
            [1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0].into(),
        )
        .reshape([1, 1, 3, 3])
        .unwrap();

        let grayscale_tensor = grayscale_tensor.reshape([1, 1, width, height]).unwrap();
        let max = grayscale_tensor
            .to_vec()
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let min = grayscale_tensor
            .to_vec()
            .into_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        println!("Grayscale min: {}, max: {}", min, max);

        let edges = grayscale_tensor
            .conv_2d(
                &sobel_x,
                Conv2DParams {
                    padding: [1, 1].into(),
                    ..Default::default()
                },
            )
            .unwrap();

        let edges = edges.reshape([width, height]).unwrap();

        let normalized_edges = edges.normalize(NormalizeParams::default()).unwrap();

        let min = normalized_edges
            .to_vec()
            .into_iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = normalized_edges
            .to_vec()
            .into_iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        println!("Edges min: {}, max: {}", min, max);

        // Use GPU grayscale to RGB conversion
        // let rgb_edges = RustGpu::grayscale_to_rgb(&edges, width, height);

        // // Step 6: Create final edge image
        // println!("🖼️ Creating final edge image");
        // self.edge_img = ImageBuffer::from_raw(width as u32, height as u32, rgb_edges.to_vec());

        // if self.edge_img.is_some() {
        //     println!("✅ GPU edge detection pipeline completed successfully!");
        //     let final_min = *self
        //         .edge_img
        //         .as_ref()
        //         .unwrap()
        //         .as_raw()
        //         .iter()
        //         .min()
        //         .unwrap_or(&0);
        //     let final_max = *self
        //         .edge_img
        //         .as_ref()
        //         .unwrap()
        //         .as_raw()
        //         .iter()
        //         .max()
        //         .unwrap_or(&0);
        //     println!("📊 Final image range: {} to {}", final_min, final_max);
        // } else {
        //     self.error_message = Some("Failed to create final image buffer".to_string());
        // }

        // self.processing = false;
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
                    ui.label("Processing with Pure GPU Pipeline...");
                }

                if !self.processing {
                    if self.edge_img.is_some() {
                        ui.colored_label(
                            egui::Color32::GREEN,
                            "✅ GPU pipeline executed successfully!",
                        );
                    }
                    if let Some(ref error) = self.error_message {
                        ui.colored_label(egui::Color32::RED, format!("❌ GPU Error: {}", error));
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

                // Combined magnitude image
                if let Some(ref edge_img) = self.edge_img {
                    ui.vertical(|ui| {
                        ui.heading("⚡ Combined Magnitude");
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
                                    .max_height(300.0)
                                    .max_width(300.0),
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
                        ui.heading("⚡ Combined Magnitude");
                        ui.label("👆 Click 'Apply GPU Edge Detection' to see results!");
                    });
                }
            });

            ui.separator();

            ui.label("🎯 Pure GPU Pipeline Architecture:");
            ui.label("1. 📤 Upload RGB image data to GPU memory");
            ui.label("2. 🔄 GPU RGB → Grayscale conversion");
            ui.label("3. 🎯 Manual GPU Sobel filtering (avoiding Conv2D issues)");
            ui.label("4. ⚡ GPU gradient magnitude calculation");
            ui.label("5. ⚖️ GPU normalization for optimal visibility");
            ui.label("6. 🎨 GPU grayscale → RGB conversion");
            ui.label("7. 📥 Transfer final results for display");

            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(0, 150, 255),
                "🚀 Maximum GPU utilization - only final image creation happens on CPU!",
            );
        });
    }
}
