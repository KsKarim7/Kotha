import os
import threading

import numpy as np

from kotha.dialogue.routing import Language

VOICES: dict[Language, tuple[str, str]] = {
    "bn": ("facebook/mms-tts-ben", "0da99de6074c8829121cdabfbdba423af18e8e56"),
    "en": ("facebook/mms-tts-eng", "c71de0fe7204c83f1c10820a7d696d0b450048ba"),
}
SEED = 0
# the seed is global to torch, so one synthesis at a time keeps every voice reproducible
SYNTHESIS_LOCK = threading.Lock()


class MmsVoice:
    def __init__(self, lang: Language) -> None:
        os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
        # imported here so that code using a fake voice does not load torch
        from transformers import AutoTokenizer, VitsModel

        model_id, revision = VOICES[lang]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        self.model = VitsModel.from_pretrained(model_id, revision=revision, local_files_only=True)
        self.model.eval()
        self.sampling_rate: int = self.model.config.sampling_rate

    def synthesize(self, text: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer(text, return_tensors="pt")
        with SYNTHESIS_LOCK, torch.inference_mode():
            torch.manual_seed(SEED)
            waveform = self.model(**inputs).waveform
        return waveform[0].numpy().astype(np.float32)
