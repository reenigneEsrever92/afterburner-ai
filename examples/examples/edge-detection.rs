use eframe::egui as ef;
use egui::{ColorImage, TextureOptions};
use image::{EncodableLayout, ImageBuffer, Rgb};

fn main() {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "My egui App",
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
