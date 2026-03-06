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
    raw_reflection: str
    reflection_sanitized: bool
    draft_image: Image.Image
    refined_image: Image.Image
    refine_prompt: str
    refine_input_images: list[Image.Image]
    used_edit_mode: bool


class ReflectionRegenerationPipeline:
    def __init__(self, image_backend, reflector) -> None:
        self.image_backend = image_backend
        self.reflector = reflector

    @staticmethod
    def _normalize_reflection(reflection: str) -> tuple[str, bool]:
        text = str(reflection or "").strip()
        original = text

        # Make problem/fix tags minimally well-formed for downstream prompting.
        if "<problem>" in text and "</problem>" not in text:
            insert_at = text.find("<fix>")
            if insert_at == -1:
                insert_at = text.find("</fix>")
            if insert_at == -1:
                text = f"{text}</problem>"
            else:
                text = f"{text[:insert_at]}</problem>{text[insert_at:]}"

        if "<fix>" not in text and "</fix>" in text:
            end_idx = text.find("</fix>")
            text = f"{text[:end_idx]}<fix>{text[end_idx:]}"

        if "<fix>" in text and "</fix>" not in text:
            text = f"{text}</fix>"

        if "<|refl_start|>" not in text:
            text = f"<|refl_start|>{text}"
        if "<|refl_end|>" not in text:
            text = f"{text}<|refl_end|>"

        return text, (text != original)

    @staticmethod
    def _align_refine_images(images: list[Image.Image], target_size: tuple[int, int]) -> list[Image.Image]:
        aligned: list[Image.Image] = []
        for img in images:
            if img.size == target_size:
                aligned.append(img)
            else:
                aligned.append(img.resize(target_size, Image.BICUBIC))
        return aligned

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

        raw_reflection = self.reflector.reflect(prompt=prompt, image=draft)
        reflection, reflection_sanitized = self._normalize_reflection(raw_reflection)
        refine_prompt = self.build_refine_prompt(prompt=prompt, reflection=reflection)

        refine_images = [draft] if init_image is None else [init_image, draft]
        refine_images = self._align_refine_images(refine_images, draft.size)
        used_edit_mode = False
        if hasattr(self.image_backend, "generate_edit"):
            used_edit_mode = True
            refined = self.image_backend.generate_edit(
                prompt=refine_prompt,
                images=refine_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        else:
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
            refine_input_images=refine_images,
            used_edit_mode=used_edit_mode,
            raw_reflection=raw_reflection,
            reflection_sanitized=reflection_sanitized,
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
        refine_debug_path = out_dir / "debug_refine.json"
        refine_input_dir = out_dir / "debug_refine_inputs"

        save_image(result.draft_image, draft_path)
        save_image(result.refined_image, refined_path)
        refine_input_paths: list[str] = []
        for idx, img in enumerate(result.refine_input_images):
            input_path = refine_input_dir / f"input_{idx}.png"
            save_image(img, input_path)
            refine_input_paths.append(str(input_path))
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
        save_json(
            refine_debug_path,
            {
                "prompt": result.prompt,
                "raw_reflection": result.raw_reflection,
                "reflection": result.reflection,
                "reflection_sanitized": result.reflection_sanitized,
                "refine_prompt": result.refine_prompt,
                "used_edit_mode": result.used_edit_mode,
                "num_refine_input_images": len(result.refine_input_images),
                "refine_input_images": refine_input_paths,
                "refine_input_specs": [
                    {"index": i, "mode": img.mode, "size": [int(img.width), int(img.height)]}
                    for i, img in enumerate(result.refine_input_images)
                ],
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "seed": seed,
                "backend_last_call_debug": getattr(self.image_backend, "last_call_debug", None),
            },
        )

        return {
            "draft": str(draft_path),
            "refined": str(refined_path),
            "meta": str(meta_path),
            "refine_debug": str(refine_debug_path),
            "reflection": result.reflection,
        }
