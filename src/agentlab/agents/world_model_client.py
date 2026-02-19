"""HTTP client for the Emu3.5 world model server."""

import base64
import io
import logging
import time

import numpy as np
import requests
from PIL import Image

logger = logging.getLogger(__name__)


class WorldModelClient:
    """Sends screenshots + candidate actions to the Emu3.5 server, returns predictions."""

    def __init__(self, server_url: str, mode: str = "image", timeout: int = 600):
        self.server_url = server_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout

    def _encode_screenshot(self, screenshot: np.ndarray) -> str:
        img = Image.fromarray(screenshot)
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _decode_image(self, b64: str) -> np.ndarray:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return np.array(img)

    def predict_batch(self, screenshot: np.ndarray, actions: list[str]) -> list[dict]:
        """Send screenshot + actions to server, return list of prediction dicts.

        Processes each action via async job queue (submit → poll → retrieve).
        Each dict has keys:
            - "action": str
            - "image": np.ndarray or None (if mode=="image")
            - "text": str or None (if mode=="text")
        """
        screenshot_b64 = self._encode_screenshot(screenshot)
        results = []

        for idx, action in enumerate(actions, 1):
            logger.info(f"[WM Client] Processing action {idx}/{len(actions)}: {action[:80]}...")

            # Step 1: Submit job
            submit_payload = {
                "screenshot_b64": screenshot_b64,
                "action": action,
                "mode": self.mode,
            }
            resp = requests.post(
                f"{self.server_url}/predict/submit",
                json=submit_payload,
                timeout=10  # Short timeout for submission
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            logger.info(f"[WM Client] Job submitted: {job_id}")

            # Step 2: Poll for completion
            poll_interval = 10  # seconds
            max_wait = self.timeout  # Overall timeout (default 600s)
            start_time = time.time()

            while time.time() - start_time < max_wait:
                time.sleep(poll_interval)

                status_resp = requests.get(
                    f"{self.server_url}/predict/status/{job_id}",
                    timeout=10
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()

                if status_data["status"] == "completed":
                    logger.info(f"[WM Client] Job {job_id} completed")
                    break
                elif status_data["status"] == "failed":
                    error = status_data.get("error", "Unknown error")
                    raise RuntimeError(f"World model job {job_id} failed: {error}")
                else:
                    # Still processing
                    logger.info(f"[WM Client] Job {job_id} status: {status_data['status']}")
            else:
                raise TimeoutError(f"World model job {job_id} timed out after {max_wait}s")

            # Step 3: Retrieve result
            result_resp = requests.get(
                f"{self.server_url}/predict/result/{job_id}",
                timeout=30  # Allow time for result retrieval
            )
            result_resp.raise_for_status()
            pred = result_resp.json()

            # Decode result
            entry = {"action": pred["action"], "image": None, "text": None}
            if pred.get("result_b64"):
                entry["image"] = self._decode_image(pred["result_b64"])
            if pred.get("result_text"):
                entry["text"] = pred["result_text"]
            results.append(entry)

            logger.info(f"[WM Client] Action {idx}/{len(actions)} completed")

        return results

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=10)
            return resp.status_code == 200 and resp.json().get("model_loaded", False)
        except Exception:
            return False
