use core::panic;
use std::env;

use audio::input::{load_audio, log_mel_spectrogram};
use model::load::load_model;
use model::tokenizer::{get_encoding};


// Todo could upgrade to clap CLI - perhaps for server deployment?
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut debug = false;

    if args.len() >= 5 {
        debug = true;
        println!("Debug enabled!");
    }
    else if args.len() < 4 {
        panic!(
            "Need to supply model configuration (.config), model tensor weights (.safetensor), and lastly audio file input (.wav)"
        );
    } 

    let model = load_model(&args[1], &args[2], None)?;

   
    // load audio input
    let audio = load_audio(&args[3], None)?;
    let log_mel_spectrogram_input = log_mel_spectrogram(audio, None, None, None)?;

    let result = model.forward_encoder(log_mel_spectrogram_input);

    println!("Forward Encoder pass done, {:?}", result);

    println!("Getting Tokenzier");
    let tokenizer = get_encoding(None, None)?;
  

    // encode_with_special_tokens returns a Vec of tokens, so we grab the first one [0]
    let end_of_text_id = tokenizer.encode_with_special_tokens("<|endoftext|>")[0];
    let start_of_transcript_id = tokenizer.encode_with_special_tokens("<|startoftranscript|>")[0];
    let english_language_id = tokenizer.encode_with_special_tokens("<|en|>")[0];
    let transcribe_task_id = tokenizer.encode_with_special_tokens("<|transcribe|>")[0];

    // Prime your decoder sequence dynamically
    let mut tokens = vec![
        start_of_transcript_id,
        english_language_id,
        transcribe_task_id,
    ];

    println!("Starting decoder loop");
    for _ in 0..448 {
        // Pass the audio features and your growing token list to the decoder
        // TODO: really shouldn't clone here need to fix params to better accept by ref not value
        let next_token = model.forward_decoder_from_token_vector(&tokens, result.clone());
        

        // Stop if Whisper says it's done
        if next_token == end_of_text_id {
            break;
        }

        tokens.push(next_token);

        if debug {
            let text_tokens: Vec<u32> = tokens
                    .clone()
                    .into_iter()
                    .filter(|&t| t < end_of_text_id) 
                    .collect();  
            let decoded_text = tokenizer.decode(&text_tokens)?;
            println!("DEBUG: {}", decoded_text);
        }
    }

    let text_tokens: Vec<u32> = tokens
        .into_iter()
        .filter(|&t| t < end_of_text_id) 
        .collect();

    let decoded_text = tokenizer.decode(&text_tokens)?;
    println!("Result: {}", decoded_text);

    Ok(())
}
