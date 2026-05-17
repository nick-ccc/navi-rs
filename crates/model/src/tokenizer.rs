use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::hash::BuildHasherDefault;

use tiktoken_rs::{CoreBPE};
use base64::{prelude::BASE64_STANDARD, Engine};
use rustc_hash::FxHasher;


type TiktokenMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

pub const LANGUAGES: &[(&str, &str)] = &[
    ("en", "english"),
    ("zh", "chinese"),
    ("de", "german"),
    ("es", "spanish"),
    ("ru", "russian"),
    ("ko", "korean"),
    ("fr", "french"),
    ("ja", "japanese"),
    ("pt", "portuguese"),
    ("tr", "turkish"),
    ("pl", "polish"),
    ("ca", "catalan"),
    ("nl", "dutch"),
    ("ar", "arabic"),
    ("sv", "swedish"),
    ("it", "italian"),
    ("id", "indonesian"),
    ("hi", "hindi"),
    ("fi", "finnish"),
    ("vi", "vietnamese"),
    ("he", "hebrew"),
    ("uk", "ukrainian"),
    ("el", "greek"),
    ("ms", "malay"),
    ("cs", "czech"),
    ("ro", "romanian"),
    ("da", "danish"),
    ("hu", "hungarian"),
    ("ta", "tamil"),
    ("no", "norwegian"),
    ("th", "thai"),
    ("ur", "urdu"),
    ("hr", "croatian"),
    ("bg", "bulgarian"),
    ("lt", "lithuanian"),
    ("la", "latin"),
    ("mi", "maori"),
    ("ml", "malayalam"),
    ("cy", "welsh"),
    ("sk", "slovak"),
    ("te", "telugu"),
    ("fa", "persian"),
    ("lv", "latvian"),
    ("bn", "bengali"),
    ("sr", "serbian"),
    ("az", "azerbaijani"),
    ("sl", "slovenian"),
    ("kn", "kannada"),
    ("et", "estonian"),
    ("mk", "macedonian"),
    ("br", "breton"),
    ("eu", "basque"),
    ("is", "icelandic"),
    ("hy", "armenian"),
    ("ne", "nepali"),
    ("mn", "mongolian"),
    ("bs", "bosnian"),
    ("kk", "kazakh"),
    ("sq", "albanian"),
    ("sw", "swahili"),
    ("gl", "galician"),
    ("mr", "marathi"),
    ("pa", "punjabi"),
    ("si", "sinhala"),
    ("km", "khmer"),
    ("sn", "shona"),
    ("yo", "yoruba"),
    ("so", "somali"),
    ("af", "afrikaans"),
    ("oc", "occitan"),
    ("ka", "georgian"),
    ("be", "belarusian"),
    ("tg", "tajik"),
    ("sd", "sindhi"),
    ("gu", "gujarati"),
    ("am", "amharic"),
    ("yi", "yiddish"),
    ("lo", "lao"),
    ("uz", "uzbek"),
    ("fo", "faroese"),
    ("ht", "haitian creole"),
    ("ps", "pashto"),
    ("tk", "turkmen"),
    ("nn", "nynorsk"),
    ("mt", "maltese"),
    ("sa", "sanskrit"),
    ("lb", "luxembourgish"),
    ("my", "myanmar"),
    ("bo", "tibetan"),
    ("tl", "tagalog"),
    ("mg", "malagasy"),
    ("as", "assamese"),
    ("tt", "tatar"),
    ("haw", "hawaiian"),
    ("ln", "lingala"),
    ("ha", "hausa"),
    ("ba", "bashkir"),
    ("jw", "javanese"),
    ("su", "sundanese"),
    ("yue", "cantonese"),
];

pub fn get_encoding(
    num_languages: Option<usize>,
    name: Option<&str>,
) -> anyhow::Result<CoreBPE> {

    let name = name.unwrap_or("gpt2");
    let num_languages = num_languages.unwrap_or(1);


    // store tiktoken file in root_dir/assets/
    let vocab_path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "assets", &format!("{}.tiktoken", name)]
        .iter()
        .collect();

    println!("Vocab path: {:?}", vocab_path);
    let file = File::open(vocab_path)?;

    let reader = BufReader::new(file);
    let mut encoder: TiktokenMap<Vec<u8>, u32> = HashMap::default();
    
    for line in reader.lines() {
        let line = line?;
        if line.is_empty(){
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() !=2 {
             return Err(anyhow::anyhow!("Invalid line format in tiktoken file"));
        }

        let token_bytes = BASE64_STANDARD.decode(parts[0])?;
        let rank: u32 = parts[1].parse()?;

        encoder.insert(token_bytes, rank);
    }

    let mut n_vocab = encoder.len();
    let mut next_vocab = || { n_vocab += 1; n_vocab as u32 };

    let mut special_tokens: TiktokenMap<String, u32>= HashMap::default();

    // Add first two tokens
    special_tokens.insert(String::from("<|endoftext|>"), next_vocab());
    special_tokens.insert(String::from("<|startoftranscript|>"), next_vocab());

    for lan in LANGUAGES.iter().take(num_languages) {
        let token_name = format!("<|{}|>", lan.1);
        special_tokens.insert(token_name, next_vocab());
    }

    special_tokens.insert(String::from("<|translate|>"), next_vocab());
    special_tokens.insert(String::from("<|transcribe|>"), next_vocab());
    special_tokens.insert(String::from("<|startofml|>"), next_vocab());
    special_tokens.insert(String::from("<|startofprev|>"), next_vocab());
    special_tokens.insert(String::from("<|nospeech|>"), next_vocab());
    special_tokens.insert(String::from("<|notimestamps|>"), next_vocab());

    for i in 0..=1500 {
        // 1. Rust requires explicit casting for mixed-type math (i as f64)
        // 2. format!("{:.2}", ...) ensures exactly 2 decimal places
        let token_name = format!("<|{:.2}|>", i as f64 * 0.02);
        
        special_tokens.insert(token_name, next_vocab());
    }
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    let bpe = CoreBPE::new(encoder, special_tokens, pattern)?;
    Ok(bpe)
}

