"""Draft -> reflection -> refine inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from qwen_latent_cot.prompts import EDITING_SYSTEM_PROMPT
from qwen_latent_cot.utils import save_image, save_json


@dataclass
class PipelineResult:
    prompt: str
    reflection: str
    draft_image: Image.Image
    refined_image: Image.Image
    refine_prompt: str


class ReflectionRegenerationPipeline:
    def __init__(self, image_backend, reflector) -> None:
        self.image_backend = image_backend
        self.reflector = reflector

    def build_refine_prompt(self, prompt: str, reflection: str) -> str:
        return (
            f"{EDITING_SYSTEM_PROMPT}\n"
            f"Goal: {prompt}\n"
            f"Feedback: {reflection}\n"
            "Regenerate the image to satisfy the goal with the feedback."
        )

    def run(
        self,
        prompt: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        init_image: Image.Image | None = None,
    ) -> PipelineResult:
        draft = self.image_backend.generate(
            prompt=prompt,
            image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        reflection = self.reflector.reflect(prompt=prompt, image=draft)
        refine_prompt = self.build_refine_prompt(prompt=prompt, reflection=reflection)

        refined = self.image_backend.generate(
            prompt=refine_prompt,
            image=draft,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return PipelineResult(
            prompt=prompt,
            reflection=reflection,
            draft_image=draft,
            refined_image=refined,
            refine_prompt=refine_prompt,
        )

    def run_and_save(
        self,
        prompt: str,
        output_dir: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        seed: int | None = None,
        init_image: Image.Image | None = None,
    ) -> dict:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result = self.run(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            init_image=init_image,
        )

        draft_path = out_dir / "draft.png"
        refined_path = out_dir / "refined.png"
        meta_path = out_dir / "result.json"

        save_image(result.draft_image, draft_path)
        save_image(result.refined_image, refined_path)
        save_json(
            meta_path,
            {
                "prompt": result.prompt,
                "reflection": result.reflection,
                "refine_prompt": result.refine_prompt,
                "draft_image": str(draft_path),
                "refined_image": str(refined_path),
            },
        )

        return {
            "draft": str(draft_path),
            "refined": str(refined_path),
            "meta": str(meta_path),
            "reflection": result.reflection,
        }
