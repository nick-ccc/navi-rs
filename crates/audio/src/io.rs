// Audio I/O
// Ingest WAV files
//
// Portions of this code are adapted from:
// https://github.com/TrevorS/voxtral-mini-realtime-rs
// Author: TrevorS
// License: Apache

use anyhow::{Context, Result};
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    Left = -1,
    Both = 0,
    Right = 1,
}

impl AudioBuffer {
    // Create buffer
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    // Create new buffer that is empty
    pub fn empty(sample_rate: u32) -> Self {
        Self {
            samples: Vec::new(),
            sample_rate,
        }
    }

    // check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    // Get the duration
    pub fn get_duration_s(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    // Normalize autio to peak amplitude - scaling all values
    pub fn scale_amplitude(&mut self, peak: f32) -> Result<()> {
        let max_amp = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        if max_amp < 1e-10 {
            // warning / error here
            anyhow::bail!(
                "Scaling can not be applied, \
                current audio signal is extremely low \
                (max amplitude = {})",
                max_amp
            );
        }

        let scale = peak / max_amp;
        for s in &mut self.samples {
            *s *= scale;
        }
        Ok(())
    }

    // Pad sample to target size and with target value
    pub fn pad(&mut self, target_size: usize, target_value: f32, axis: PadSide) -> Result<()> {
        let current_length = self.samples.len();

        if current_length >= target_size {
            anyhow::bail!(
                "Current audio sample size is already, or exceeds \
                desired target size."
            )
        }

        let pad_amount = target_size - current_length;
        match axis {
            PadSide::Left => {
                let mut new_samples = vec![target_value; pad_amount];
                new_samples.extend_from_slice(&self.samples);
                self.samples = new_samples;
            }
            PadSide::Right => {
                self.samples.extend(vec![target_value; pad_amount]);
            }
            PadSide::Both => {
                let left_pad = pad_amount / 2;
                let right_pad = pad_amount - left_pad;
                let mut new_samples = vec![target_value; left_pad];
                new_samples.extend_from_slice(&self.samples);
                new_samples.extend(vec![target_value; right_pad]);
                self.samples = new_samples;
            }
        }

        // process successful
        Ok(())
    }

    // Trim to size
    pub fn trim(&mut self, target_size: usize, axis: PadSide) -> Result<()> {
        let current_length = self.samples.len();

        if current_length <= target_size {
            anyhow::bail!(
                "Current audio size is already smaller than or equal to \
                target size"
            )
        }

        let trim_amount = current_length - target_size;
        match axis {
            PadSide::Left => {
                self.samples = self.samples[trim_amount..].to_vec();
            }
            PadSide::Right => {
                self.samples.truncate(target_size);
            }
            PadSide::Both => {
                let left_trim = trim_amount / 2;
                let right_trim = trim_amount - left_trim + current_length;
                self.samples = self.samples[left_trim..right_trim].to_vec();
            }
        }

        Ok(())
    }

    pub fn append(&mut self, other: &AudioBuffer) -> Result<()> {
        if self.sample_rate != other.sample_rate {
            anyhow::bail!(
                "Sample rate mismatch between self ({}) and target ({})",
                self.sample_rate,
                other.sample_rate
            );
        }
        self.samples.extend_from_slice(&other.samples);
        Ok(())
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_wav(self, path)
    }
}

pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<AudioBuffer> {
    let path = path.as_ref();
    let reader = WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;

            reader
                .into_samples::<i32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read WAV samples")?
                .chunks(channels)
                .map(|chunk| {
                    // Mix to mono by averaging channels
                    let sum: i32 = chunk.iter().sum();
                    (sum as f32 / channels as f32) / max_val
                })
                .collect()
        }
        SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .context("Failed to read WAV samples")?
                .chunks(channels)
                .map(|chunk| {
                    // Mix to mono by averaging channels
                    chunk.iter().sum::<f32>() / channels as f32
                })
                .collect()
        }
    };

    Ok(AudioBuffer::new(samples, sample_rate))
}

pub fn save_wav<P: AsRef<Path>>(audio: &AudioBuffer, path: P) -> Result<()> {
    let path = path.as_ref();
    let spec = WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create WAV file: {}", path.display()))?;

    for &sample in &audio.samples {
        // Clamp and convert to i16
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer.write_sample(i16_sample)?;
    }

    writer.finalize()?;
    Ok(())
}
