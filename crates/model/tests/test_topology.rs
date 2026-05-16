use burn::tensor::{Int, Tensor};
use burn_wgpu::{Wgpu, WgpuDevice};
use model::model::{ModelDimensions, Whisper};

type B = Wgpu<f32>;

/// Centralized model configurations using the Tiny English variant parameters
const DIMENSIONS: ModelDimensions = ModelDimensions {
    n_mels: 80,
    n_audio_ctx: 1500,
    n_audio_state: 384,
    n_audio_head: 6,
    n_audio_layer: 4,
    n_vocab: 51864, // English-only variant vocabulary size
    n_text_ctx: 448,
    n_text_state: 384,
    n_text_head: 6,
    n_text_layer: 4,
};

/// Whisper audio input length before convolutional downsampling (30 seconds @ 10ms hops)
const INPUT_TIME_FRAMES: usize = 3000;

fn setup() -> (Whisper<B>, WgpuDevice) {
    let device = WgpuDevice::default();
    let model = Whisper::<B>::new(&DIMENSIONS, &device);
    (model, device)
}

// Encoder Test Suite
fn run_encoder_test(batch_size: usize) {
    let (model, device) = setup();

    let mel = Tensor::<B, 3>::zeros([batch_size, DIMENSIONS.n_mels, INPUT_TIME_FRAMES], &device);
    let out = model.forward_encoder(mel);

    assert_eq!(
        out.dims(),
        [batch_size, DIMENSIONS.n_audio_ctx, DIMENSIONS.n_audio_state]
    );
}

#[test]
fn whisper_forward_encoder_single_batch() {
    run_encoder_test(1);
}

#[test]
fn whisper_forward_encoder_large_batch() {
    run_encoder_test(30);
}

// Decoder Test Suite
fn run_decoder_test(batch_size: usize, sequence_length: usize) {
    let (model, device) = setup();

    let tokens = Tensor::<B, 2, Int>::zeros([batch_size, sequence_length], &device);
    let encoder_output = Tensor::<B, 3>::zeros(
        [batch_size, DIMENSIONS.n_audio_ctx, DIMENSIONS.n_audio_state],
        &device,
    );

    let out = model.forward_decoder(tokens, encoder_output);

    assert_eq!(
        out.dims(),
        [batch_size, sequence_length, DIMENSIONS.n_vocab]
    );
}

#[test]
fn whisper_forward_decoder_single_batch_full_context() {
    run_decoder_test(1, DIMENSIONS.n_text_ctx);
}

#[test]
fn whisper_forward_decoder_large_batch_full_context() {
    run_decoder_test(30, DIMENSIONS.n_text_ctx);
}

#[test]
fn whisper_forward_decoder_autoregressive_step() {
    // Verifies that the model successfully handles small incremental
    // sequence slices (like predicting a single token during generation)
    run_decoder_test(1, 1);
}
