<h1 align="center">NAVI STT in Rust</h1>
<p align="center">
  <a href="https://zelda.fandom.com/wiki/Navi">
    <img 
        src="https://static.wikia.nocookie.net/zelda_gamepedia_en/images/0/08/OoT3D_Navi_Artwork.png/revision/latest?cb=20110729222747"  
        alt="Navi" 
        width="400" 
    />
  </a>
</p>


<p align="center">
  <a href="(https://github.com/nick-ccc/navi-rs/actions/workflows/rust.yaml">
    <img src="https://github.com/nick-ccc/navi-rs/actions/workflows/rust.yaml/badge.svg" alt="website"/>
  </a>
  <a href="https://codecov.io/gh/nick-ccc/navi-rs">
    <img src="https://codecov.io/gh/nick-ccc/navi-rs/branch/main/graph/badge.svg" alt="website"/>
  </a>
  
</p>


# Purpose:
This is purely for learning and deciphering the architecture of the Whsiper model from OpenAI and is by no means an 
original creation. 

# Status
The model is capable of only producing the token corresponding to, manslaughter (for current test input file).

# Refences

> [!NOTE]
> Much of this projects works on the ML Speech to Text is reimplementation from the below project

- Original Whipser Paper: https://github.com/openai/whisper/tree/main
- Implementation from other rust burn project: https://github.com/Gadersd/whisper-burn/tree/main
    - Particular parts have been forked, mainly the audio logic for MEL spectogram and parts of the model, 
        with some updates to reflect later burn versions (0.20)
