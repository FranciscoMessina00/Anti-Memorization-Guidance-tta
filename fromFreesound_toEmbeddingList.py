"""End-to-end Freesound -> local dataset -> CLAP embedding -> embeddings JSON appender.

Workflow per id (from input.csv):
1. Skip if already present in embeddings_new.json with an 'embedding'.
2. Ensure audio file downloaded to soundDataset/sound_<id>.wav using Freesound download API.
3. Fetch description field (if not already stored) from Freesound metadata endpoint.
4. Compute CLAP embedding.
5. Append/update entry in embeddings_new.json as:
   {
	 "<id>": {
	   "embedding": [...],
	   "conditioning": {"prompt": <description>, "seconds_start": 0, "seconds_total": <duration_seconds>}
	 }
   }

Supports resuming safely: writes JSON after each successful id.

Starting point options:
* Provide --start-id <id> on the command line to begin from a specific id present in the CSV (first match).
* Or set environment variable START_SOUND_ID to the desired id (command line flag overrides env var).
If the id isn't found, the script logs a warning and processes all ids (or the first id numerically >= provided numeric value).

Fill ACCESS_TOKEN below (OAuth2 bearer) or set FREESOUND_ACCESS_TOKEN env var.
You can also provide an API token (non-OAuth) via FREESOUND_API_TOKEN; the script auto-selects header format.
"""

from __future__ import annotations

import os
import csv
import json
import time
import math
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

import requests
import librosa
from tqdm import tqdm

try:
	HERE = os.path.dirname(__file__)
	ROOT = os.path.abspath(os.path.join(HERE, '..'))
	LOCAL_CLAP = os.path.join(ROOT, 'CLAP', 'src')
	import sys
	sys.path.insert(0, LOCAL_CLAP)
	import laion_clap  # type: ignore
except ImportError as e:  # pragma: no cover
	raise SystemExit("Could not import laion_clap. Ensure submodule/package is available.") from e

# ---------------- Configuration ---------------- #
INPUT_CSV = Path("../input.csv").resolve()
EMBEDDINGS_JSON = Path("embeddings_new.json").resolve()
SOUND_FOLDER = Path("../soundDataset").resolve()
SOUND_FOLDER.mkdir(parents=True, exist_ok=True)

ACCESS_TOKEN = os.getenv("FREESOUND_ACCESS_TOKEN", "")
API_TOKEN = os.getenv("FREESOUND_API_TOKEN", "")
if not ACCESS_TOKEN and not API_TOKEN:
	ACCESS_TOKEN = ""  # manual fill optional

RATE_LIMIT_SLEEP = 1.2
DOWNLOAD_RETRY = 3
TIMEOUT = 60
SAVE_EVERY = 1

# ---------------- Logging ---------------- #
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("freesound_pipeline")


def load_embeddings(path: Path) -> Dict[str, Any]:
	if path.exists() and path.stat().st_size > 0:
		try:
			with path.open("r", encoding="utf-8") as f:
				return json.load(f)
		except json.JSONDecodeError:
			logger.warning("embeddings JSON corrupted; starting with empty dict (backup created)")
			backup = path.with_suffix(".corrupt.bak")
			path.rename(backup)
	return {}


def save_embeddings(data: Dict[str, Any], path: Path) -> None:
	tmp = path.with_suffix(".tmp")
	with tmp.open("w", encoding="utf-8") as f:
		json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
	tmp.replace(path)


def already_has_embedding(entry: Any) -> bool:
	return isinstance(entry, dict) and "embedding" in entry and entry.get("embedding")


def download_sound(sound_id: str, dest_path: Path) -> bool:
	if dest_path.exists():
		return True
	if not (ACCESS_TOKEN or API_TOKEN):
		logger.error("No Freesound token provided. Set FREESOUND_ACCESS_TOKEN or FREESOUND_API_TOKEN.")
		return False
	url = f"https://freesound.org/apiv2/sounds/{sound_id}/download/"
	headers = {"Accept": "application/json"}
	if ACCESS_TOKEN:
		headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
	else:
		headers["Authorization"] = f"Token {API_TOKEN}"
	for attempt in range(1, DOWNLOAD_RETRY + 1):
		try:
			r = requests.get(url, headers=headers, timeout=TIMEOUT)
			if r.status_code == 401:
				logger.error("Unauthorized (401) for id %s. Check token type.", sound_id)
				return False
			if r.ok:
				dest_path.parent.mkdir(parents=True, exist_ok=True)
				with dest_path.open("wb") as f:
					f.write(r.content)
				return True
			else:
				logger.warning("Download failed (%s) for id %s: %s", r.status_code, sound_id, r.text[:200])
		except requests.RequestException as e:
			logger.warning("Attempt %d download error for %s: %s", attempt, sound_id, e)
		time.sleep(1.5 * attempt)
	return False


def fetch_description(sound_id: str) -> str:
	if not (ACCESS_TOKEN or API_TOKEN):
		return ""
	url = f"https://freesound.org/apiv2/sounds/{sound_id}/?fields=description"
	headers = {"Accept": "application/json"}
	if ACCESS_TOKEN:
		headers["Authorization"] = f"Bearer {ACCESS_TOKEN}"
	else:
		headers["Authorization"] = f"Token {API_TOKEN}"
	try:
		r = requests.get(url, headers=headers, timeout=TIMEOUT)
		if r.ok:
			return r.json().get("description", "")
		else:
			logger.warning("Metadata failed (%s) for id %s", r.status_code, sound_id)
	except requests.RequestException as e:
		logger.warning("Metadata request error for %s: %s", sound_id, e)
	return ""


def compute_duration(audio_path: Path) -> float:
	try:
		y, sr = librosa.load(audio_path, sr=None, mono=True)
		return float(librosa.get_duration(y=y, sr=sr))
	except Exception as e:
		logger.warning("Failed to compute duration for %s: %s", audio_path.name, e)
		return 0.0


def init_clap() -> Any:
	logger.info("Loading CLAP model ...")
	model = laion_clap.CLAP_Module(enable_fusion=False)
	model.load_ckpt()
	logger.info("CLAP loaded")
	return model


def compute_embedding(clap_model: Any, audio_path: Path) -> list[float]:
	emb = clap_model.get_audio_embedding_from_filelist(x=[str(audio_path)], use_tensor=False)
	vec = emb[0]
	return [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]


def process(start_id: str | None = None):
	if not INPUT_CSV.exists():
		raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

	embeddings = load_embeddings(EMBEDDINGS_JSON)
	clap_model = init_clap()

	with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f, delimiter=';')
		if 'id' not in (reader.fieldnames or []):
			raise SystemExit("CSV missing 'id' column")
		ids = [row['id'] for row in reader if row.get('id')]

	if start_id:
		try:
			idx = ids.index(start_id)
			if idx > 0:
				logger.info("Starting from id %s (skipping %d earlier ids)", start_id, idx)
			ids = ids[idx:]
		except ValueError:
			if start_id.isdigit():
				target = int(start_id)
				cut = [i for i, sid in enumerate(ids) if sid.isdigit() and int(sid) >= target]
				if cut:
					logger.info("Start id %s not found; starting from first id >= it (id %s)", start_id, ids[cut[0]])
					ids = ids[cut[0]:]
				else:
					logger.warning("Start id %s not found and no higher numeric ids; processing all.")
			else:
				logger.warning("Start id %s not found; processing all ids.")

	processed = 0
	for sound_id in tqdm(ids, desc="Processing sounds"):
		entry = embeddings.get(sound_id)
		if already_has_embedding(entry):
			continue

		audio_path = SOUND_FOLDER / f"sound_{sound_id}.wav"
		if not download_sound(sound_id, audio_path):
			logger.error("Skipping %s due to download failure", sound_id)
			continue

		description = ""
		if isinstance(entry, dict):
			description = entry.get('conditioning', {}).get('prompt', '') or ""
		if not description:
			description = fetch_description(sound_id)
			time.sleep(RATE_LIMIT_SLEEP)

		try:
			embedding_vector = compute_embedding(clap_model, audio_path)
		except Exception as e:
			logger.error("Embedding failed for %s: %s", sound_id, e)
			continue

		duration_sec = compute_duration(audio_path)
		if math.isnan(duration_sec):
			duration_sec = 0.0

		embeddings[sound_id] = {
			'embedding': embedding_vector,
			'conditioning': {
				'prompt': description,
				'seconds_start': 0,
				'seconds_total': duration_sec
			}
		}

		processed += 1
		if processed % SAVE_EVERY == 0:
			save_embeddings(embeddings, EMBEDDINGS_JSON)
			logger.info(f"Saved progress ({processed} processed), id: {sound_id}")

	save_embeddings(embeddings, EMBEDDINGS_JSON)
	logger.info(f"Done. Total newly processed: {processed}, last id: {sound_id}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Freesound -> CLAP embeddings pipeline")
	parser.add_argument("--start-id", dest="start_id", type=str, default=None,
						help="Begin processing from this sound id (first occurrence) or closest higher numeric id if exact not found.")
	args = parser.parse_args()
	start_id_env = os.getenv("START_SOUND_ID")
	start_id = args.start_id or start_id_env
	process(start_id=start_id)
