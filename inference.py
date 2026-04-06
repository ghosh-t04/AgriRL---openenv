"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local image to use for the environment if using from_docker_image()

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import re
import textwrap
import requests
from typing import List, Optional

from client import AgricultureAction, AgricultureEnv
from dotenv import load_dotenv

load_dotenv()

# =========================
# ENV / CONFIG
# =========================

HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("IMAGE_NAME")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"


TASK_NAME = os.getenv("AGRICULTURE_TASK", "crop-selection")
BENCHMARK = os.getenv("AGRICULTURE_BENCHMARK", "agriculture")

MAX_STEPS = 3
TEMPERATURE = 0.3
MAX_TOKENS = 80
SUCCESS_SCORE_THRESHOLD = 0.55

# Approx max reward ~ 54 per step
MAX_TOTAL_REWARD = MAX_STEPS * 54.0

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set. Please set your Hugging Face token.")

if not IMAGE_NAME:
    raise ValueError("IMAGE_NAME is not set. Please set your Docker image name.")

# =========================
# PROMPT
# =========================

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an agricultural decision-making agent optimizing for maximum reward.

    You will receive a farm state containing:
    - soil type
    - nitrogen, phosphorus, potassium
    - pH
    - rainfall
    - temperature
    - humidity
    - groundwater
    - season

    Your task is to choose the SINGLE best crop for highest agricultural suitability and sustainability.

    Crop guidance:
    - rice: best for high rainfall, high humidity, water-rich conditions
    - wheat: best for cool/moderate temperature and moderate rainfall
    - maize: best for balanced nutrients and moderate climate
    - cotton: best in black soil, warm temperatures, moderate water
    - groundnut: best in sandy/loamy soil, lower pH, moderate rainfall
    - pulses: best in lower nitrogen soil and dry/moderate climate
    - millet: best for dry, low-rainfall, hardy conditions
    - sugarcane: best for high water, warm temperature, fertile soil

    Important:
    - Prioritize matching rainfall, season, soil type, and pH strongly.
    - Avoid unsuitable crops even if they are generally popular.
    - Choose only one crop.

    Reply with ONLY one crop name from:
    rice, wheat, maize, cotton, groundnut, pulses, millet, sugarcane

    No explanation.
    """
).strip()

# =========================
# LOGGING
# =========================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# =========================
# MODEL HELPERS
# =========================

def build_user_prompt(obs) -> str:
    return textwrap.dedent(
        f"""
        Current farm state:
        soil_type: {obs.soil_type}
        nitrogen: {obs.nitrogen}
        phosphorus: {obs.phosphorus}
        potassium: {obs.potassium}
        ph: {obs.ph}
        rainfall: {obs.rainfall}
        temperature: {obs.temperature}
        humidity: {obs.humidity}
        groundwater: {obs.groundwater}
        season: {obs.season}

        Choose the best crop.
        """
    ).strip()


def query_huggingface(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }

    response = requests.post(
        f"{API_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API Error {response.status_code}: {response.text}")

    result = response.json()

    return result["choices"][0]["message"]["content"].strip()


def extract_crop(text: str) -> str:
    text = text.strip().lower()
    crops = ["rice", "wheat", "maize", "cotton", "groundnut", "pulses", "millet", "sugarcane"]

    for crop in crops:
        if re.search(rf"\b{crop}\b", text):
            return crop

    return "millet"  # safe fallback


def get_model_crop(obs) -> str:
    user_prompt = build_user_prompt(obs)

    try:
        raw_text = query_huggingface(user_prompt)
        print(f"[DEBUG] Raw model output: {raw_text}", flush=True)
        return extract_crop(raw_text)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "millet"

# =========================
# MAIN LOOP
# =========================

async def main() -> None:
    env = await AgricultureEnv.from_docker_image('agriculture_env')

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            crop = get_model_crop(obs)

            result = await env.step(AgricultureAction(crop_name=crop))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=crop, reward=reward, done=done, error=error)

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())