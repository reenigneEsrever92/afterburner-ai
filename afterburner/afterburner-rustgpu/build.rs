use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let current_dir = std::env::current_dir()?;
    let mut build_dir = current_dir.clone();
    build_dir.push("../../shaders");
    let build_dir = build_dir.canonicalize()?;

    println!("Current dir: {:?}, build dir: {:?}", current_dir, build_dir);
    println!("Env: {:?}", std::env::vars());

    let mut cargo = Command::new("cargo");

    for (key, _) in std::env::vars() {
        if key.contains("CARGO") || key.contains("RUST") {
            cargo.env_remove(key);
        }
    }

    let output = cargo
        .current_dir(build_dir)
        .args(["run", "-r", "-p", "afterburner-ai-builder"])
        .output()?;

    if !output.status.success() {
        eprintln!(
            "Builder failed with stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(code) = output.status.code() {
            std::process::exit(code);
        } else {
            std::process::exit(1);
        }
    }

    // Parse the output to find the cargo:rustc-env line and forward it
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.starts_with("cargo:rustc-env=") {
            println!("{}", line);
        }
    }

    Ok(())
}
