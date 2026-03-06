from pathlib import Path

from PIL import Image

from qwen_latent_cot.inference.pipeline import ReflectionRegenerationPipeline
from qwen_latent_cot.models.qwen_image_backend import MockQwenImageBackend
from qwen_latent_cot.models.reflector import HeuristicReflector


def test_mock_pipeline_runs_and_saves(tmp_path: Path):
    pipeline = ReflectionRegenerationPipeline(
        image_backend=MockQwenImageBackend(width=256, height=256),
        reflector=HeuristicReflector(),
    )

    result = pipeline.run_and_save(
        prompt="A minimal poster with geometric shapes",
        output_dir=str(tmp_path),
        num_inference_steps=5,
        guidance_scale=1.0,
        seed=123,
    )

    assert Path(result["draft"]).exists()
    assert Path(result["refined"]).exists()
    assert Path(result["meta"]).exists()
    assert "<|refl_start|>" in result["reflection"]


class _DummyEditBackend:
    def __init__(self):
        self.called_with_images = None

    def generate(self, prompt, image=None, num_inference_steps=50, guidance_scale=4.0, seed=None):
        del prompt, image, num_inference_steps, guidance_scale, seed
        return Image.new("RGB", (64, 64), (10, 20, 30))

    def generate_edit(self, prompt, images, num_inference_steps=50, guidance_scale=4.0, seed=None):
        del prompt, num_inference_steps, guidance_scale, seed
        self.called_with_images = list(images)
        return Image.new("RGB", (64, 64), (40, 50, 60))


def test_refine_uses_generate_edit_when_available(tmp_path: Path):
    backend = _DummyEditBackend()
    pipeline = ReflectionRegenerationPipeline(
        image_backend=backend,
        reflector=HeuristicReflector(),
    )
    init = Image.new("RGB", (64, 64), (1, 2, 3))

    result = pipeline.run_and_save(
        prompt="A tiny test prompt",
        output_dir=str(tmp_path),
        num_inference_steps=3,
        guidance_scale=1.0,
        seed=7,
        init_image=init,
    )

    assert backend.called_with_images is not None
    assert len(backend.called_with_images) == 2
    assert Path(result["refined"]).exists()
