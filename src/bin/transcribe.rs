use core::panic;
use std::env;

use audio::input::{load_audio, log_mel_spectrogram};
use model::load::load_model;

// Todo could upgrade to clap CLI - perhaps for server deployment?

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        panic!(
            "Need to supply model configuration (.config), model tensor weights (.safetensor), and lastly audio file input (.wav)"
        );
    }

    let model = load_model(&args[1], &args[2], None)?;

    // load audio input
    let mut audio = load_audio(&args[3], None)?;
    let log_mel_spectrogram_input = log_mel_spectrogram(audio, None, None, None)?;

    let result = model.forward_encoder(log_mel_spectrogram_input);

    println!("Forward Encoder pass done, {:?}", result);

    Ok(())
}
