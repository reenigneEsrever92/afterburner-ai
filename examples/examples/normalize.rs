use afterburner_rustgpu::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the RustGPU backend
    init();

    println!("=== Afterburner Normalize Operation Examples ===\n");

    // Example 1: L2 Normalization (default, similar to PyTorch F.normalize)
    println!("1. L2 Normalization (Unit Vector):");
    let input_1d: Tensor<RustGpu, 1, f32> =
        RustGpu::new_tensor(Shape([4]), vec![3.0, 4.0, 0.0, 0.0]);

    println!("Input: {:?}", input_1d.to_vec());

    let normalized_1d = input_1d.normalize(NormalizeParams::default())?;
    let result_1d = normalized_1d.to_vec();

    println!("L2 Normalized: {:?}", result_1d);

    // Verify it's a unit vector (norm should be 1.0)
    let norm: f32 = result_1d.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("Resulting norm: {:.6} (should be ~1.0)\n", norm);

    // Example 2: L2 Normalization along different dimensions (2D tensor)
    println!("2. L2 Normalization on 2D tensor (row-wise):");
    let input_2d: Tensor<RustGpu, 2, f32> = RustGpu::new_tensor(
        Shape([2, 3]),
        vec![
            1.0, 2.0, 3.0, // Row 1: [1, 2, 3]
            4.0, 5.0, 6.0,
        ], // Row 2: [4, 5, 6]
    );

    println!("Input matrix (2x3):");
    let input_vals = input_2d.to_vec();
    println!(
        "  Row 0: [{:.1}, {:.1}, {:.1}]",
        input_vals[0], input_vals[1], input_vals[2]
    );
    println!(
        "  Row 1: [{:.1}, {:.1}, {:.1}]",
        input_vals[3], input_vals[4], input_vals[5]
    );

    // Normalize along dimension 1 (each row becomes unit vector)
    let normalized_2d = input_2d.normalize((2.0, 1))?; // L2 norm, dim=1
    let result_2d = normalized_2d.to_vec();

    println!("L2 Normalized (each row is unit vector):");
    println!(
        "  Row 0: [{:.4}, {:.4}, {:.4}]",
        result_2d[0], result_2d[1], result_2d[2]
    );
    println!(
        "  Row 1: [{:.4}, {:.4}, {:.4}]",
        result_2d[3], result_2d[4], result_2d[5]
    );

    // Verify each row is a unit vector
    let norm_row0 = (result_2d[0].powi(2) + result_2d[1].powi(2) + result_2d[2].powi(2)).sqrt();
    let norm_row1 = (result_2d[3].powi(2) + result_2d[4].powi(2) + result_2d[5].powi(2)).sqrt();
    println!(
        "Row 0 norm: {:.6}, Row 1 norm: {:.6}\n",
        norm_row0, norm_row1
    );

    // Example 3: L1 Normalization (Manhattan norm)
    println!("3. L1 Normalization (Manhattan distance):");
    let input_l1: Tensor<RustGpu, 1, f32> =
        RustGpu::new_tensor(Shape([4]), vec![1.0, 2.0, 3.0, 4.0]);

    println!("Input: {:?}", input_l1.to_vec());

    let normalized_l1 = input_l1.normalize(NormalizeParams {
        p: 1.0, // L1 norm
        dim: 0,
        eps: 1e-12,
    })?;
    let result_l1 = normalized_l1.to_vec();

    println!("L1 Normalized: {:?}", result_l1);

    // Verify L1 norm is 1.0
    let l1_norm: f32 = result_l1.iter().map(|x| x.abs()).sum();
    println!("Resulting L1 norm: {:.6} (should be ~1.0)\n", l1_norm);

    // Example 4: Column-wise normalization (dim=0)
    println!("4. Column-wise L2 Normalization (dim=0):");
    let input_col: Tensor<RustGpu, 2, f32> = RustGpu::new_tensor(
        Shape([3, 2]),
        vec![
            1.0, 4.0, // Row 0: [1, 4]
            2.0, 5.0, // Row 1: [2, 5]
            3.0, 6.0,
        ], // Row 2: [3, 6]
    );

    println!("Input matrix (3x2):");
    let input_col_vals = input_col.to_vec();
    for i in 0..3 {
        println!(
            "  Row {}: [{:.1}, {:.1}]",
            i,
            input_col_vals[i * 2],
            input_col_vals[i * 2 + 1]
        );
    }

    // Normalize along dimension 0 (each column becomes unit vector)
    let normalized_col = input_col.normalize(NormalizeParams::default())?; // L2 norm, dim=0
    let result_col = normalized_col.to_vec();

    println!("Column-wise normalized:");
    for i in 0..3 {
        println!(
            "  Row {}: [{:.4}, {:.4}]",
            i,
            result_col[i * 2],
            result_col[i * 2 + 1]
        );
    }

    // Verify each column is a unit vector
    let col0_norm = (result_col[0].powi(2) + result_col[2].powi(2) + result_col[4].powi(2)).sqrt();
    let col1_norm = (result_col[1].powi(2) + result_col[3].powi(2) + result_col[5].powi(2)).sqrt();
    println!(
        "Column 0 norm: {:.6}, Column 1 norm: {:.6}\n",
        col0_norm, col1_norm
    );

    // Example 5: Using negative dimension indexing
    println!("5. Negative dimension indexing (dim=-1 equals dim=1 for 2D):");
    let input_neg: Tensor<RustGpu, 2, f32> = RustGpu::new_tensor(
        Shape([2, 3]),
        vec![
            1.0, 0.0, 0.0, // Row 0: [1, 0, 0] (already unit vector)
            0.0, 3.0, 4.0,
        ], // Row 1: [0, 3, 4] (3-4-5 triangle)
    );

    println!("Input matrix (2x3):");
    let input_neg_vals = input_neg.to_vec();
    println!(
        "  Row 0: [{:.1}, {:.1}, {:.1}]",
        input_neg_vals[0], input_neg_vals[1], input_neg_vals[2]
    );
    println!(
        "  Row 1: [{:.1}, {:.1}, {:.1}]",
        input_neg_vals[3], input_neg_vals[4], input_neg_vals[5]
    );

    let normalized_neg = input_neg.normalize((2.0, -1))?; // dim=-1 means last dimension
    let result_neg = normalized_neg.to_vec();

    println!("Normalized with dim=-1:");
    println!(
        "  Row 0: [{:.4}, {:.4}, {:.4}]",
        result_neg[0], result_neg[1], result_neg[2]
    );
    println!(
        "  Row 1: [{:.4}, {:.4}, {:.4}]",
        result_neg[3], result_neg[4], result_neg[5]
    );

    println!("\n=== Summary ===");
    println!("The normalize operation is similar to PyTorch's F.normalize:");
    println!("- Default: L2 normalization (p=2.0) along dim=1");
    println!("- Supports different p-norms (L1, L2, etc.)");
    println!("- Can normalize along any dimension");
    println!("- Negative dimension indexing works (-1 = last dim)");
    println!("- Prevents division by zero with epsilon parameter");
    println!("- Each vector along the specified dimension becomes unit length");

    Ok(())
}
