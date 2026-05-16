use core::panic;
use std::env;

use model::load::load_model;

// Todo could upgrade to clap CLI - perhaps for server deployment?

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        panic!(
            "Need to supply model configuration (.config) and model tensor weights (.safetensor)"
        );
    }

    let model = load_model(&args[1], &args[2], None)?;

    Ok(())
}
