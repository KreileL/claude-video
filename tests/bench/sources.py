"""Pinned model sources of the pt-BR benchmark (fetch_models.py → install_models.py).

License of each model checked on its Hugging Face card on 2026-10-18.
"""

# Fine-tunes converted to CTranslate2 here, installed as watch-mcp/<name>:
# (source repo, pinned revision, local name, license)
CONVERT = [
    # Brazilian Portuguese: Common Voice 16 + a private pt-BR set (model card).
    ("freds0/distil-whisper-large-v3-ptbr", "3edf997edaa2a34c233d6f0d0184d9a33ef4132a",
     "distil-whisper-large-v3-ptbr-ct2", "mit"),
    # Portuguese Common Voice 11 (mostly Brazilian speakers).
    ("pierreguillou/whisper-medium-portuguese", "9ca8ecf81d8e5207570784145ef5586c645a1264",
     "whisper-medium-portuguese-ct2", "apache-2.0"),
]
CONVERT_FILES = [
    "config.json", "generation_config.json", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
    "special_tokens_map.json", "normalizer.json", "model.safetensors", "pytorch_model.bin",
]

# Ready-made speaches models, installed under their own id: (repo, revision, license, files)
DIRECT = [
    ("deepdml/faster-whisper-large-v3-turbo-ct2", "4df90f75321148c3a29a9e2351b7ddf8f5b115a8", "mit",
     ["README.md", "config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]),
    # Not a candidate: only builds the approximate references of the real clips.
    ("Systran/faster-whisper-large-v3", "edaa852ec7e145841d8ffdb056a99866b5f0a478", "mit",
     ["README.md", "config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"]),
    # TTS for the synthetic corpus (Kokoro-82M is Apache-2.0 upstream; Brazilian voices pf_dora, pm_alex, pm_santa).
    ("speaches-ai/Kokoro-82M-v1.0-ONNX-int8", "f1637656d821e888ea08230d5cc140777d16edec", "apache-2.0",
     ["README.md", "model.onnx", "voices.bin"]),
    ("speaches-ai/piper-pt_BR-edresson-low", "77a0fd77f2ed335db8abdb55e61380418259ebef", "see card",
     ["README.md", "config.json", "model.onnx"]),
]
