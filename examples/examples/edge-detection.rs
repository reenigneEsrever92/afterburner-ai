use std::io::Read;

use afterburner_rustgpu::prelude::*;
use eframe::egui as ef;
use egui::{ColorImage, TextureOptions};
use image::{EncodableLayout, ImageBuffer, Rgb};

fn main() {
    init();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Image Viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .expect("Running GUI");
}

struct App {
    img: ImageBuffer<Rgb<u8>, Vec<u8>>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);

        let img = image::ImageReader::open("assets/stones.jpg")
            .expect("Image loaded")
            .decode()
            .expect("Image decoded")
            .to_rgb8();

        App { img }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &ef::Context, _frame: &mut eframe::Frame) {
        let bytes = self.img.to_vec();
        let t: Tensor<RustGpu, 1, u8> = bytes.into();
        let t: Tensor<_, 1, f32> = t.convert();

        let image = ColorImage::from_rgb(
            [self.img.width() as _, self.img.height() as _],
            self.img.as_bytes(),
        );
        ef::CentralPanel::default().show(ctx, |ui| {
            let texture = ctx.load_texture("img", image, TextureOptions::default());
            ui.add(
                egui::Image::new(&texture)
                    .corner_radius(5)
                    .max_height(500.0)
                    .max_width(500.0),
            );
        });
    }
}
