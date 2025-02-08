// use clap::Parser;
use spirv_builder::SpirvBuilder;

// #[derive(Parser, Debug)]
// #[command(version, about, long_about = None)]
// struct Args {
//     #[arg(required = true)]
//     path: String,
// }

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // println!("In process vars: {:?}", std::env::vars());
    // let args = Args::parse();
    // std::env::set_current_dir(args.path);
    SpirvBuilder::new("shaders", "spirv-unknown-vulkan1.1").build()?;
    Ok(())
}
