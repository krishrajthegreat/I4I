# -*- coding: utf-8 -*-
"""
pipeline/roboflow_workflow.py
==============================
Roboflow Workflow REST client.

Calls the Roboflow serverless workflow endpoint directly via HTTPS POST,
passing an image and an expected_class label per call.

Workflow details:
  Workspace  : k-krish-raj
  Workflow ID: gemini-machine-instance-auto-label-2
  Endpoint   : https://serverless.roboflow.com/k-krish-raj/workflows/
               gemini-machine-instance-auto-label-2

Declared inputs:
  - image          : image file path or URL
  - expected_class : string label (e.g. "band_saw", "cnc_milling")

Usage (standalone)
------------------
  python pipeline/roboflow_workflow.py --image path/to/image.jpg --class band_saw
  python pipeline/roboflow_workflow.py --image https://example.com/img.jpg --class milling

Usage (as a module)
-------------------
  from pipeline.roboflow_workflow import run_workflow
  result = run_workflow(image="path/to/image.jpg", expected_class="band_saw")
  print(result)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Endpoint config
# ---------------------------------------------------------------------------
WORKSPACE_SLUG    = "kkr-r543n"
WORKFLOW_ID       = "gemini-machine-instance-auto-label-2"
ENDPOINT_URL      = (
    f"https://serverless.roboflow.com/{WORKSPACE_SLUG}"
    f"/workflows/{WORKFLOW_ID}"
)
REQUEST_TIMEOUT   = 60   # seconds


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def run_workflow(
    image: str,
    expected_class: str,
    api_key: str | None = None,
) -> dict:
    """
    Call the Roboflow workflow endpoint with an image and expected class.

    Parameters
    ----------
    image : str
        Either:
          - A local file path (e.g. "dataset/train/band_saw/img_001.jpg")
          - A publicly accessible image URL (e.g. "https://...")
    expected_class : str
        The class label string to pass as the workflow's expected_class input
        (e.g. "band_saw", "cnc_milling", "panel_saw").
    api_key : str, optional
        Roboflow API key. Falls back to ROBOFLOW_API_KEY env variable.

    Returns
    -------
    dict
        Parsed JSON response from the workflow endpoint.

    Raises
    ------
    ValueError
        If API key is missing or the image path does not exist.
    requests.HTTPError
        If the HTTP request fails (non-2xx response).
    """
    _api_key = api_key or os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not _api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not set. Add it to .env or pass api_key= explicitly."
        )

    # Build the image payload
    image_payload: dict
    img_path = Path(image)
    if img_path.exists() and img_path.is_file():
        # Local file — encode as base64
        raw = img_path.read_bytes()
        b64  = base64.b64encode(raw).decode("utf-8")
        suffix = img_path.suffix.lower().lstrip(".")
        mime   = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png",  "webp": "image/webp",
        }.get(suffix, "image/jpeg")
        image_payload = {"type": "base64", "value": f"data:{mime};base64,{b64}"}
    elif image.startswith("http://") or image.startswith("https://"):
        # URL — pass directly
        image_payload = {"type": "url", "value": image}
    else:
        raise ValueError(
            f"Image must be a local file path or an http(s) URL. Got: {image!r}"
        )

    payload = {
        "api_key": _api_key,
        "inputs": {
            "image": image_payload,
            "expected_class": expected_class,
        },
    }

    response = requests.post(
        ENDPOINT_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def run_workflow_batch(
    image_paths: list[str],
    expected_class: str,
    api_key: str | None = None,
) -> list[dict]:
    """
    Call run_workflow() for a list of images and return all results.
    Failed calls are logged but do not abort the batch.
    """
    results = []
    for i, img in enumerate(image_paths, start=1):
        try:
            result = run_workflow(img, expected_class, api_key)
            results.append({"image": img, "status": "ok", "response": result})
            print(f"  [{i}/{len(image_paths)}] {Path(img).name} -> OK")
        except Exception as e:
            results.append({"image": img, "status": "error", "error": str(e)})
            print(f"  [{i}/{len(image_paths)}] {Path(img).name} -> FAILED: {e}")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call the Roboflow Gemini auto-label workflow via REST."
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH_OR_URL",
        help="Local image file path or https:// URL.",
    )
    parser.add_argument(
        "--class",
        dest="expected_class",
        required=True,
        metavar="LABEL",
        help="Expected class label (e.g. band_saw, cnc_milling).",
    )
    args = parser.parse_args()

    print(f"Calling Roboflow workflow:")
    print(f"  Endpoint      : {ENDPOINT_URL}")
    print(f"  Image         : {args.image}")
    print(f"  Expected class: {args.expected_class}")
    print()

    try:
        result = run_workflow(args.image, args.expected_class)
        print("Response:")
        print(json.dumps(result, indent=2))
    except requests.HTTPError as e:
        print(f"[HTTP ERROR] {e.response.status_code}: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
