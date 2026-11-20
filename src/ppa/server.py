"""HTTP REST API server for PPA model inference.

Provides endpoints for health check, single image analysis, and comparative analysis:
  - GET  /api/health
  - POST /api/analyze
  - POST /api/compare

Zero third-party web framework dependencies (built with Python standard library
ThreadingHTTPServer) to preserve existing pyproject.toml and uv.lock management.
"""

from __future__ import annotations

import argparse
import email
import json
import logging
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Tuple

from .config import REPO_ROOT
from .inference import DEFAULT_CKPT_PATH, PPAInferenceEngine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ppa.server")


def parse_multipart(content_type: str, body: bytes) -> Tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Parse multipart/form-data payload into fields and files."""
    fields: dict[str, str] = {}
    files: dict[str, dict[str, Any]] = {}

    header = f"Content-Type: {content_type}\r\n\r\n".encode("latin1")
    msg = email.message_from_bytes(header + body)

    for part in msg.walk():
        if part.is_multipart():
            continue
        cd = part.get("Content-Disposition", "")
        if not cd:
            continue

        params: dict[str, str] = {}
        for item in cd.split(";"):
            item = item.strip()
            if "=" in item:
                k, v = item.split("=", 1)
                params[k.strip().lower()] = v.strip().strip('"')

        name = params.get("name")
        filename = params.get("filename")
        payload = part.get_payload(decode=True)

        if not name or payload is None:
            continue

        if filename:
            files[name] = {"filename": filename, "data": payload}
        else:
            fields[name] = payload.decode("utf-8", errors="replace")

    return fields, files


class PPAHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for PPA inference API."""

    engine: PPAInferenceEngine

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, Accept")

    def _send_json_response(self, data: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

    def _send_error_response(self, message: str, status: int = HTTPStatus.BAD_REQUEST) -> None:
        self._send_json_response({"error": message, "status": status}, status=status)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        """Handle GET requests (health checks)."""
        clean_path = self.path.split("?")[0]
        if clean_path in ("/api/health", "/health", "/"):
            self._send_json_response({
                "online": True,
                "status": "healthy",
                "device": self.engine.device_name,
                "checkpoint": self.engine.checkpoint_path.name,
                "best_epoch": self.engine.epoch,
                "val_metrics": self.engine.val_metrics,
            })
        else:
            self._send_error_response(f"Endpoint not found: {clean_path}", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Handle POST requests (inference and comparison)."""
        clean_path = self.path.split("?")[0]

        # Read content length
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            self._send_error_response("Invalid Content-Length header")
            return

        if content_length <= 0:
            self._send_error_response("Empty request body")
            return

        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")

        if clean_path in ("/api/analyze", "/analyze"):
            self._handle_analyze(content_type, body)
        elif clean_path in ("/api/compare", "/compare"):
            self._handle_compare(content_type, body)
        else:
            self._send_error_response(f"Endpoint not found: {clean_path}", HTTPStatus.NOT_FOUND)

    def _handle_analyze(self, content_type: str, body: bytes) -> None:
        """Handle single image aesthetic evaluation."""
        image_data = None

        if "multipart/form-data" in content_type:
            _, files = parse_multipart(content_type, body)
            for key in ("image", "file", "photo"):
                if key in files:
                    image_data = files[key]["data"]
                    break
            if image_data is None and files:
                first_key = next(iter(files))
                image_data = files[first_key]["data"]
        elif content_type.startswith("image/"):
            image_data = body

        if not image_data:
            self._send_error_response("No image file provided in form-data ('image' or 'file')")
            return

        try:
            result = self.engine.predict_image(image_data)
            logger.info(
                f"Analyzed image -> rawScore={result['rawScore']}, "
                f"tier='{result['tier']}', latency={result['inferenceTimeMs']}ms"
            )
            self._send_json_response(result)
        except Exception as exc:
            logger.exception("Error during image analysis")
            self._send_error_response(f"Inference failure: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_compare(self, content_type: str, body: bytes) -> None:
        """Handle comparative analysis between two images."""
        if "multipart/form-data" not in content_type:
            self._send_error_response("Comparison requires multipart/form-data")
            return

        _, files = parse_multipart(content_type, body)

        img_a = None
        img_b = None

        # Look for imageA / imageB or image1 / image2
        for k in ("imageA", "image_a", "image1", "imgA"):
            if k in files:
                img_a = files[k]["data"]
                break
        for k in ("imageB", "image_b", "image2", "imgB"):
            if k in files:
                img_b = files[k]["data"]
                break

        # Fallback: take first two files
        if (img_a is None or img_b is None) and len(files) >= 2:
            keys = list(files.keys())
            img_a = files[keys[0]]["data"]
            img_b = files[keys[1]]["data"]

        if not img_a or not img_b:
            self._send_error_response("Comparison requires two images ('imageA' and 'imageB')")
            return

        try:
            res_a = self.engine.predict_image(img_a)
            res_b = self.engine.predict_image(img_b)

            score_diff = abs(res_a["rawScore"] - res_b["rawScore"])
            is_a_winner = res_a["rawScore"] >= res_b["rawScore"]

            response_payload = {
                "resultA": res_a,
                "resultB": res_b,
                "isAWinner": is_a_winner,
                "winner": "Photo A" if is_a_winner else "Photo B",
                "scoreDiff": round(score_diff, 2),
            }
            logger.info(
                f"Compared images -> Winner: {response_payload['winner']} "
                f"(A: {res_a['rawScore']}, B: {res_b['rawScore']}, diff: {score_diff:.2f})"
            )
            self._send_json_response(response_payload)
        except Exception as exc:
            logger.exception("Error during comparative analysis")
            self._send_error_response(f"Comparison failure: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        """Route standard HTTP server logging to standard logger."""
        logger.debug(f"{self.address_string()} - {format % args}")


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    checkpoint_path: Path | str = DEFAULT_CKPT_PATH,
    device: str | None = None,
) -> None:
    """Initialize PPA inference engine and start HTTP server."""
    logger.info(f"Initializing PPA Inference Engine from: {checkpoint_path}")
    engine = PPAInferenceEngine(checkpoint_path=checkpoint_path, device=device)
    logger.info(
        f"PPA Model loaded successfully! Device: {engine.device_name}, "
        f"Epoch: {engine.epoch}, Validation SRCC: {engine.val_metrics.get('srcc', 'N/A')}"
    )

    PPAHTTPRequestHandler.engine = engine

    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, PPAHTTPRequestHandler)
    logger.info(f"PPA Inference API server running on http://{host}:{port}")
    logger.info("Ready to accept inference requests at /api/analyze and /api/compare")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        httpd.server_close()
        logger.info("Server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PPA Inference API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT_PATH), help="Path to checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint, device=args.device)


if __name__ == "__main__":
    main()
