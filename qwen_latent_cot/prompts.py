"""Centralized prompts used by training and inference."""

REFLECTION_SYSTEM_PROMPT = (
    "You are a visual quality analyst. Given a generated image and the user's goal, "
    "identify concrete problems and provide actionable fixes. "
    "Respond in this exact format:\n"
    "<observation>...analysis and actionable fixes...</observation>"
)

EDITING_SYSTEM_PROMPT = (
    "You are an image editor. Given an image and observation/fix text, regenerate an improved image "
    "that better satisfies the goal while preserving unrelated details."
)

STAGE11_SYSTEM_PROMPT = "You are a helpful assistant."

STAGE11_ROUND1_PROMPT = (
    "Based on the image and the editing goal, analyze what needs to be changed "
    "and provide editing instructions."
)

STAGE11_ROUND2_PROMPT = (
    "Based on the original image and previous edit result, analyze whether the goal is achieved "
    "and provide further editing instructions."
)
