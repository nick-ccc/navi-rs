use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::Result;
use burn::tensor::backend::Backend;
use burn_store::ModuleSnapshot;
use burn_wgpu::Wgpu;
use serde::Deserialize;

use crate::model::{ModelDimensions, Whisper};

type B = Wgpu<f32>;

// Converts hf config to struct we can then read into model configs
// Would realistically have been more ideal to load straight into model configs
// bug HF names are not super clear on what becomes what in model
#[derive(Debug, Deserialize)]
struct HuggingFaceWhisperConfig {
    pub num_mel_bins: usize, // n_mels
    pub d_model: usize,      // n_audio_state and n_text_state

    // encoder
    pub max_source_positions: usize,    // n_audio_ctx
    pub encoder_attention_heads: usize, // n_audio_head
    pub encoder_layers: usize,          // n_audio_layer

    // decoder
    pub max_target_positions: usize,    // n_text_ctx
    pub decoder_attention_heads: usize, // n_text_head
    pub decoder_layers: usize,          // n_text_layer

    // token embeddings
    pub vocab_size: usize, // n_vocab
}

fn load_whisper_dimensions<P: AsRef<Path>>(path: P) -> Result<ModelDimensions> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // Deserializes using the exact JSON fields from your file
    let hf_config: HuggingFaceWhisperConfig = serde_json::from_reader(reader)?;

    Ok(ModelDimensions {
        n_mels: hf_config.num_mel_bins,
        n_audio_ctx: hf_config.max_source_positions,
        n_audio_state: hf_config.d_model,
        n_audio_head: hf_config.encoder_attention_heads,
        n_audio_layer: hf_config.encoder_layers,
        n_vocab: hf_config.vocab_size,
        n_text_ctx: hf_config.max_target_positions,
        n_text_state: hf_config.d_model,
        n_text_head: hf_config.decoder_attention_heads,
        n_text_layer: hf_config.decoder_layers,
    })
}

pub fn load_model(
    model_config_path: &str,
    safte_tensor_path: &str,
    device: Option<<Wgpu as Backend>::Device>,
) -> Result<Whisper<B>> {
    let resolved_device = device.unwrap_or_else(|| <Wgpu as Backend>::Device::default());

    // Load model dimensions here
    print!("Loading model dimensions\n");
    let dims = load_whisper_dimensions(model_config_path)?;
    let model = Whisper::<B>::new(&dims, &resolved_device);

    print!("Loading safetensors\n");
    let mut store = burn_store::SafetensorsStore::from_file(safte_tensor_path).overwrite(true);
    model.save_into(&mut store)?;

    // load model here
    Ok(model)
}
