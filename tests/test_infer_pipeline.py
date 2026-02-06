from pathlib import Path

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
    assert "observation" in result["reflection"]
