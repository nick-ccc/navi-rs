use burn::tensor::{ElementConversion, Float, Shape, Tensor, TensorData, backend::Backend};
use burn_wgpu::Wgpu;
use std::process::{Command, Stdio};

const SAMPLE_RATE: usize = 16000;

const N_MELS: usize = 80;
const ALLOWED_MELS: [usize; 2] = [80, 128];

const N_FFT: usize = 400;
const HOP_LENGTH: usize = 160;

type B = Wgpu<f32>;

pub fn load_audio(file_path: &str, sample_rate: Option<usize>) -> anyhow::Result<Vec<f32>> {
    let ar = sample_rate.unwrap_or(SAMPLE_RATE);

    let mut cmd = Command::new("ffmpeg");

    println!("Running ffmpeg to process audio");
    cmd.args([
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        &ar.to_string(),
        "-",
    ])
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr_msg = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!("Failed to load audio: {}", stderr_msg));
    }

    let audio_samples = output
        .stdout
        .chunks_exact(2)
        .map(|byte_pair| {
            // Read 2 bytes as a signed 16-bit integer (little-endian)
            let raw_sample = i16::from_le_bytes([byte_pair[0], byte_pair[1]]);

            // Cast to float and normalize to the range [-1.0, 1.0]
            raw_sample as f32 / 32768.0
        })
        .collect::<Vec<f32>>();

    Ok(audio_samples)
}

pub fn log_mel_spectrogram(
    mut audio: Vec<f32>,
    n_mels: Option<usize>,
    padding: Option<usize>,
    device: Option<<Wgpu as Backend>::Device>,
) -> anyhow::Result<Tensor<B, 3>> {
    let n_mels = n_mels.unwrap_or(N_MELS);
    let padding = padding.unwrap_or(0);
    let device = device.unwrap_or_else(|| <Wgpu as Backend>::Device::default());

    // Pad the raw vector layout
    if padding > 0 {
        audio.resize(audio.len() + padding, 0.0);
    }

    // 3. Move the contiguous vector onto the backend device as a 1D Tensor
    let total_samples = audio.len();
    let audio_tensor =
        Tensor::<B, 1>::from_data(TensorData::new(audio, Shape::new([total_samples])), &device);

    // Hann Window
    let window = hann_window_device(N_FFT, &device);

    // 4. Compute Short-Time Fourier Transform (STFT) magnitude frames
    let (stft_real, stft_imaginary) = stfft(audio_tensor.unsqueeze(), N_FFT, HOP_LENGTH, window);

    let magnitudes = stft_real.powf_scalar(2.0) + stft_imaginary.powf_scalar(2.0);
    let [n_batch, n_row, n_col] = magnitudes.dims();
    let magnitudes = magnitudes.slice([0..n_batch, 0..n_row, 0..(n_col - 1)]);

    let mel_spec = generate_mel_filters(n_mels, SAMPLE_RATE as f64, N_FFT, &device)?
        .unsqueeze()
        .matmul(magnitudes);

    // let log_spec = tensor_log10(tensor_max_scalar(mel_spec, 1.0e-10));
    // log is ln here  so we must divide to make it approximatley log10
    let log_spec = mel_spec
        .clone()
        .clamp(1.0e-10, mel_spec.max().into_scalar().elem::<f64>())
        .log()
        / 2.302585;

    let max: f64 = log_spec.clone().max().into_scalar().elem();

    let log_spec = log_spec.clamp(max - 8.0, max);
    let log_spec = (log_spec + 4.0) / 4.0;

    // 8. Final calibration scaling specific to the Whisper preprocessing spec
    println!("Log Mel Spectrogram completed.");
    Ok((log_spec + 4.0) / 4.0)
}

fn hz_to_mel_slaney(frequency: f64) -> f64 {
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0; // Linear step size (~66.67 Hz per mel)

    let min_log_hz = 1000.0;
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (6.4_f64).ln() / 27.0;

    if frequency >= min_log_hz {
        // Logarithmic scaling calculation for high-frequency spectrum
        min_log_mel + (frequency / min_log_hz).ln() / logstep
    } else {
        // Flat linear mapping step for low-frequency spectrum
        (frequency - f_min) / f_sp
    }
}

pub fn mel_to_hz_slaney_tensor<B: Backend, const D: usize>(mel: Tensor<B, D>) -> Tensor<B, D> {
    // Define Slaney parameters as element types
    let f_min = 0.0f32.elem::<B::FloatElem>();
    let f_sp = (200.0f32 / 3.0f32).elem::<B::FloatElem>();
    let min_log_hz = 1000.0f32.elem::<B::FloatElem>();
    let min_log_mel = (1000.0f32 / (200.0f32 / 3.0f32)).elem::<B::FloatElem>(); // ~15.0

    // logstep = ln(6.4) / 27.0
    let logstep = (6.4f32.ln() / 27.0f32).elem::<B::FloatElem>();

    // Formula: f_min + f_sp * mel
    let linear_hz = mel.clone() * f_sp + f_min;

    // Formula: min_log_hz * exp(logstep * (mel - min_log_mel))
    let log_hz = ((mel.clone() - min_log_mel) * logstep).exp() * min_log_hz;

    let is_linear_mask = mel.lower_elem(min_log_mel);
    let hz = log_hz.mask_where(is_linear_mask, linear_hz);

    hz
}

fn generate_fft_tensor<B: Backend>(
    n_fft: usize,
    sample_rate: f64,
    device: &B::Device,
) -> Tensor<B, 1> {
    // NOTE: technical danger and truncation here from usize to i64
    Tensor::arange(0..(n_fft / 2 + 1) as i64, device)
        .float()
        .mul_scalar(sample_rate / n_fft as f64)
}

fn generate_mel_tensors<B: Backend>(
    n_mels: usize,
    f_min: f64,
    f_max: f64,
    device: &B::Device,
) -> Tensor<B, 1> {
    let min_mel = hz_to_mel_slaney(f_min);
    let max_mel = hz_to_mel_slaney(f_max);

    let mel_bins = Tensor::arange(0..n_mels as i64, device)
        .float()
        .mul_scalar((max_mel - min_mel) / (n_mels - 1) as f64)
        .add_scalar(min_mel);

    mel_to_hz_slaney_tensor(mel_bins)
}

// MEL frequencies are expressed logarithimically from normal frequency and are meant
// to seperate pitches in a linear fashion rather than how normal frequencies notable
// pitches are exponential in distance between frequencies.
//
// n_mels == mel bands
// Expected output shape: [n_mels, N_FFT / 2]
fn generate_mel_filters<B: Backend>(
    n_mels: usize,
    sample_rate: f64,
    n_fft: usize,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, 2>> {
    if !ALLOWED_MELS.contains(&n_mels) {
        return Err(anyhow::anyhow!("Unsupported n_mels: {}", n_mels));
    }

    // Determine highest and lowest freq
    let f_min = 0.0;
    let f_max = sample_rate * 0.5; // nyquist rate

    // generate bins for fft tesnor
    let fft_tensor = generate_fft_tensor(n_fft, sample_rate, device);

    // take frequency range and linearly space into n_mels bands
    let mel_tensor = generate_mel_tensors(n_mels + 2, f_min, f_max, device);

    //np.diff equivilant
    let mel_diff =
        mel_tensor.clone().slice(1..n_mels + 2) - mel_tensor.clone().slice(0..n_mels + 1);

    //generate ramp filters
    let mel_ramps = mel_tensor
        .clone()
        .unsqueeze::<2>()
        .transpose()
        .repeat_dim(1, n_fft / 2 + 1)
        - fft_tensor.unsqueeze();

    let lower = -mel_ramps.clone().slice([0..n_mels])
        / mel_diff
            .clone()
            .slice([0..n_mels])
            .unsqueeze::<2>()
            .transpose();

    let upper = mel_ramps.slice([2..(2 + n_mels)])
        / mel_diff
            .slice([1..(1 + n_mels)])
            .unsqueeze::<2>()
            .transpose();

    // 2. Intersect lower and upper slopes (Element-wise minimum via mask_where)
    let weights = lower
        .clone()
        .mask_where(lower.clone().greater(upper.clone()), upper);

    // ... and intersect with zero (Equivalent to your ReLU step)
    let zero = 0.0f32.elem::<B::FloatElem>();
    let weights = weights.clamp_min(zero);

    // 3. Clean Slaney-style normalization directly from the mel array
    let enorm = (mel_tensor.clone().slice([2..(n_mels + 2)])
        - mel_tensor.clone().slice([0..n_mels]))
    .powf_scalar(-1.0)
        * 2.0;

    // Apply the normalization across the matrix rows
    let weights = weights * enorm.unsqueeze::<2>().transpose();

    Ok(weights)
}

pub fn hann_window_device<B: Backend>(
    window_length: usize,
    device: &B::Device,
) -> Tensor<B, 1, Float> {
    Tensor::arange(0..window_length as i64, device)
        .float()
        .mul_scalar(std::f64::consts::PI / window_length as f64)
        .sin()
        .powf_scalar(2.0)
}

/// Short time Fourier transform that takes a waveform input of size (n_batch, n_sample) and returns (real_part, imaginary_part) frequency spectrums.
/// The size of each returned tensor is (n_batch, n_freq, n_frame)
/// where n_freq = int(n_fft / 2 + 1), n_frame = int( ( n_sample_padded - n_fft ) / hop_length ) + 1,
/// n_sample_padded = if n_fft is even: n_sample + n_fft else: n_sample + n_fft - 1.
pub fn stfft<B: Backend>(
    input: Tensor<B, 2>,
    n_fft: usize,
    hop_length: usize,
    window: Tensor<B, 1>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [n_batch, orig_input_size] = input.dims();

    assert!(orig_input_size >= n_fft);

    let device = input.device();

    // add reflection padding to center the windows on the input times
    let pad = n_fft / 2;
    let left_pad = input.clone().slice([0..n_batch, 1..(pad + 1)]).flip([1]);
    let right_pad = input
        .clone()
        .slice([
            0..n_batch,
            (orig_input_size - pad - 1)..(orig_input_size - 1),
        ])
        .flip([1]);
    let input = Tensor::cat(vec![left_pad, input, right_pad], 1);

    // pad window to length n_fft
    let [orig_window_length] = window.dims();
    let window = if orig_window_length < n_fft {
        let left_pad = (n_fft - orig_window_length) / 2;
        let right_pad = n_fft - orig_window_length - left_pad;
        Tensor::cat(
            vec![
                Tensor::zeros([left_pad], &device),
                window,
                Tensor::zeros([right_pad], &device),
            ],
            0,
        )
    } else {
        window
    };

    let [_, input_size] = input.dims();

    let n_frame = (input_size - n_fft) / hop_length + 1;
    let n_freq = n_fft / 2 + 1; // assuming real input there is conjugate symmetry

    // construct matrix of overlapping input windows
    let num_parts = div_roundup(n_fft, hop_length);
    let n_hops = div_roundup(input_size, hop_length);
    let padded_input_size = n_hops * hop_length;
    let padding = Tensor::zeros([n_batch, padded_input_size - input_size], &device);
    let template = Tensor::cat(vec![input, padding], 1)
        .reshape([n_batch, n_hops, hop_length])
        .transpose();
    let parts: Vec<_> = (0..num_parts)
        .into_iter()
        .map(|i| {
            template
                .clone()
                .slice([0..n_batch, 0..hop_length, i..(n_frame + i)])
        })
        .collect();
    let input_windows = Tensor::cat(parts, 1).slice([0..n_batch, 0..n_fft, 0..n_frame]);

    // construct matrix of wave angles
    let coe = std::f64::consts::PI * 2.0 / n_fft as f64;
    let b = Tensor::arange(0..n_freq as i64, &device)
        .float()
        .mul_scalar(coe)
        .unsqueeze::<2>()
        .transpose()
        .repeat_dim(1, n_fft)
        * Tensor::arange(0..n_fft as i64, &device)
            .float()
            .unsqueeze::<2>();

    // convolve the input slices with the window and waves
    let real_part = (b.clone().cos() * window.clone().unsqueeze())
        .unsqueeze()
        .matmul(input_windows.clone());
    let imaginary_part = (b.sin() * (-window).unsqueeze())
        .unsqueeze()
        .matmul(input_windows);

    return (real_part, imaginary_part);
}

fn div_roundup(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}
