//! GPU-Accelerated Edge Detection with Channel Normalization
//!
//! This example demonstrates GPU-based edge detection using Sobel filters
//! and channel normalization to ensure edge values are properly normalized
//! between 0 and 1 for optimal visualization.
//!
//! ## Channel Normalization Usage
//!
//! This example showcases a key difference between the two normalization types:
//! - **torch.nn.functional.normalize**: Creates unit vectors (magnitude = 1)
//! - **torchvision.transforms.Normalize**: Maps values to specific range using mean/std
//!
//! For edge detection, we use **channel normalization** (torchvision style) to:
//! 1. Calculate edge gradients which have arbitrary magnitude ranges
//! 2. Map the gradient values from [min_gradient, max_gradient] to [0, 1]
//! 3. Ensure optimal visualization contrast
//!
//! The normalization formula used: `(value - min) / (max - min) = normalized_value`
//! Where min=0 and max=(max-min), giving us: `(value - min) / range`
//!
//! Features:
//! - GPU RGB to grayscale conversion
//! - Sobel edge detection using Conv2D
//! - **Channel normalization to normalize edge values to [0,1] range**
//! - Real-time processing with progress indication
//! - Side-by-side image comparison
//!
//! Pipeline:
//! 1. Load RGB image data to GPU
//! 2. GPU RGB → Grayscale conversion
//! 3. Apply Sobel X and Y filters using Conv2D
//! 4. Calculate gradient magnitude: sqrt(gx² + gy²)
//! 5. **Use channel normalization to map values to [0,1] range**
//! 6. Convert back to RGB for display

use std::time::Instant;

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

        match self.run_edge_detection() {
            Ok(edge_img) => {
                self.edge_img = Some(edge_img);
                println!("✅ GPU edge detection completed successfully!");
            }
            Err(e) => {
                self.error_message = Some(format!("Edge detection failed: {}", e));
                println!("❌ Edge detection error: {}", e);
            }
        }

        self.processing = false;
    }

    fn run_edge_detection(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>, String> {
        let width = self.original_img.width() as usize;
        let height = self.original_img.height() as usize;

        println!("🚀 Starting GPU edge detection with channel normalization");
        println!("📐 Image dimensions: {}x{}", width, height);

        // Step 1: Create RGB tensor and convert to grayscale
        let rgb_data: Vec<u8> = self.original_img.as_raw().to_vec();
        println!("📤 Uploading {} bytes to GPU", rgb_data.len());

        let rgb_tensor: Tensor<RustGpu, 1, u8> =
            RustGpu::new_tensor(Shape([width * height * 3]), rgb_data);

        println!("🔄 GPU RGB → Grayscale conversion");
        let grayscale_tensor = RustGpu::rgb_to_grayscale(&rgb_tensor, width, height);
        let grayscale_4d = grayscale_tensor.reshape([1, 1, height, width]).unwrap();

        // Step 2: Create Sobel filters
        println!("🎯 Creating Sobel filters");
        let sobel_x = RustGpu::new_tensor(
            Shape([1, 1, 3, 3]),
            vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0],
        );

        let sobel_y = RustGpu::new_tensor(
            Shape([1, 1, 3, 3]),
            vec![1.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, -2.0, -1.0],
        );

        // Step 3: Apply Sobel filters
        println!("📐 Applying Sobel X filter");
        let edges_x = grayscale_4d
            .conv_2d(
                &sobel_x,
                Conv2DParams {
                    padding: Shape([1, 1]),
                    ..Default::default()
                },
            )
            .map_err(|e| format!("Sobel X conv failed: {:?}", e))?;

        println!("📐 Applying Sobel Y filter");
        let edges_y = grayscale_4d
            .conv_2d(
                &sobel_y,
                Conv2DParams {
                    padding: Shape([1, 1]),
                    ..Default::default()
                },
            )
            .map_err(|e| format!("Sobel Y conv failed: {:?}", e))?;

        // Step 4: Calculate gradient magnitude (sqrt(gx² + gy²))
        println!("⚡ Calculating gradient magnitude");

        // Flatten for element-wise operations
        let gx_flat = edges_x.reshape([width * height]).unwrap();
        let gy_flat = edges_y.reshape([width * height]).unwrap();

        let gx_data = gx_flat.to_vec();
        let gy_data = gy_flat.to_vec();

        let magnitude_data: Vec<f32> = gx_data
            .iter()
            .zip(gy_data.iter())
            .map(|(gx, gy)| (gx * gx + gy * gy).sqrt())
            .collect();

        let magnitude_tensor = RustGpu::new_tensor(Shape([1, 1, height, width]), magnitude_data);

        // Step 5: Use channel normalization to map values to [0,1] range
        println!("⚖️ Applying channel normalization to map values to [0,1]");
        println!(
            "   This demonstrates torchvision-style normalization vs torch.functional normalize"
        );

        let start = Instant::now();

        // Calculate statistics for normalization
        // let min_val = magnitude_tensor
        //     .min(MinParams::default())
        //     .unwrap()
        //     .to_vec()
        //     .iter()
        //     .fold(f32::INFINITY, |a, &b| a.min(b));
        // let max_val = magnitude_tensor
        //     .max(MaxParams::default())
        //     .unwrap()
        //     .to_vec()
        //     .iter()
        //     .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mag_data = magnitude_tensor.to_vec();
        let min_val = mag_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = mag_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let end = Instant::now();
        let duration = end - start;

        println!("⏱️ min, max took: {:?}", duration);

        println!(
            "📊 Original magnitude range: {:.3} to {:.3}",
            min_val, max_val
        );

        // Apply channel normalization: (value - min) / (max - min) maps to [0,1]
        // This is the key difference from L2 normalization which would create unit vectors
        let range = max_val - min_val;
        let normalized_magnitude = if range > 1e-6 {
            println!(
                "   Using formula: (value - {:.3}) / {:.3} → [0,1]",
                min_val, range
            );
            magnitude_tensor
                .channel_normalize(ChannelNormalizeParams {
                    mean: vec![min_val], // Subtract minimum to shift to 0
                    std: vec![range],    // Divide by range to scale to [0,1]
                })
                .map_err(|e| format!("Channel normalization failed: {:?}", e))?
        } else {
            // Handle case where all values are the same
            println!("   All gradient values are the same - no normalization needed");
            magnitude_tensor
        };

        // Verify normalization
        let norm_data = normalized_magnitude.to_vec();
        let norm_min = norm_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let norm_max = norm_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("📊 Normalized range: {:.3} to {:.3}", norm_min, norm_max);

        // Step 6: Convert to grayscale tensor for RGB conversion
        let normalized_1d = normalized_magnitude.reshape([width * height]).unwrap();

        println!("🎨 Converting to RGB");
        let rgb_edges = RustGpu::grayscale_to_rgb(&normalized_1d, width, height);

        // Step 7: Create final image
        println!("🖼️ Creating final edge image");
        let final_data = rgb_edges.to_vec();

        ImageBuffer::from_raw(width as u32, height as u32, final_data)
            .ok_or_else(|| "Failed to create image buffer from edge data".to_string())
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

            ui.label("🎯 GPU Pipeline with Channel Normalization:");
            ui.label("1. 📤 Upload RGB image data to GPU memory");
            ui.label("2. 🔄 GPU RGB → Grayscale conversion");
            ui.label("3. 🎯 Apply Sobel X and Y filters using Conv2D");
            ui.label("4. ⚡ Calculate gradient magnitude: sqrt(gx² + gy²)");
            ui.label("5. ⚖️ Channel normalization: (value - min) / (max - min) → [0,1]");
            ui.label("   📝 This uses torchvision.transforms.Normalize style");
            ui.label("   📝 NOT torch.nn.functional.normalize (L2 unit vectors)");
            ui.label("6. 🎨 GPU grayscale → RGB conversion");
            ui.label("7. 📥 Display normalized edge results");

            ui.separator();
            ui.colored_label(
                egui::Color32::from_rgb(0, 150, 255),
                "✨ Channel normalization maps arbitrary edge gradients to [0,1] for optimal visualization!",
            );
            ui.colored_label(
                egui::Color32::from_rgb(255, 165, 0),
                "🔍 Key: Uses (value-mean)/std vs L2 normalize which creates unit vectors",
            );
        });
    }
}
