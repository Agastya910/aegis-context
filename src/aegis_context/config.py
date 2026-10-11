"""Runtime configuration, persisted at ~/.aegis/config.json."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def default_home() -> Path:
    return Path.home() / ".aegis"


@dataclass
class Config:
    home: str = field(default_factory=lambda: str(default_home()))
    capture_interval_s: float = 5.0
    ring_window_s: int = 300            # Tier-1 hot buffer: last 5 minutes
    embedding_dim: int = 384
    host: str = "127.0.0.1"             # loopback only, by design
    port: int = 8765
    ollama_url: str = "http://127.0.0.1:11434"
    vision_model: str = "moondream"
    vision_backend: str = "none"        # "none" | "ollama"
    embedder: str = "hash"              # "hash" | "onnx" | "ollama"
    embed_model: str = "nomic-embed-text"  # used when embedder == "ollama"
    onnx_model_path: str = ""
    onnx_tokenizer_path: str = ""
    digest_after_s: int = 3600          # deprecated (old rollup); kept so old config.json files load
    retention_days: int = 365           # Tier-3 digests
    raw_retention_days: int = 7         # frame screen text + chunk vectors
    episode_retention_days: int = 90

    # chat agent
    chat_backend: str = "ollama"        # "ollama" | "none"
    chat_model: str = ""                # empty = .env OLLAMA_MODEL, then answer_model, then vision_model
    chat_url: str = ""                  # empty = .env OLLAMA_BASE_URL, then ollama_url
    chat_fallback_model: str = "qwen2.5:7b"  # local, supports tools
    history_turns: int = 10
    history_char_budget: int = 12000
    agent_max_rounds: int = 4
    tool_result_chars: int = 6000

    # ingestion
    episode_gap_s: int = 180
    episode_max_s: int = 1800
    dhash_threshold: int = 8
    text_dup_threshold: float = 0.92
    vision_min_interval_s: float = 8.0
    idle_after_s: int = 120
    exclude_apps: list = field(default_factory=lambda: ["1password", "bitwarden", "keepass", "keepassxc",
                                                        "lockapp", "logonui"])
    exclude_title_patterns: list = field(default_factory=lambda: ["(?i)inprivate", "(?i)incognito",
                                                                  "(?i)private browsing"])
    chunk_size: int = 800
    # integrations (outbound; only scrubbed, generic text ever leaves the device)
    darwin_enabled: bool = True
    private_terms: list = field(default_factory=list)  # e.g. your name; scrubbed from shared context packs
    darwin_url: str = "https://api.darwin.so/api/v2"
    chunk_overlap: int = 150

    answer_backend: str = "none"        # "none" | "ollama": LLM writes the HUD/query answer
    answer_model: str = ""              # empty = same model as vision
    vision_url: str = ""                # empty = use ollama_url; .env OLLAMA_BASE_URL overrides
    # Secret: comes only from the environment / .env (OLLAMA_API_KEY), never written to config.json.
    vision_api_key: str = field(default="", repr=False, metadata={"secret": True})
    darwin_api_key: str = field(default="", repr=False, metadata={"secret": True})

    @property
    def vectors_db(self) -> Path:
        return Path(self.home) / "aegis_vectors.db"

    @classmethod
    def load(cls, home: str | Path | None = None) -> "Config":
        base = Path(home) if home else default_home()
        path = base / "config.json"
        cfg = cls(home=str(base))
        if path.exists():
            data = json.loads(path.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
            cfg.home = str(base)
        env = {**_read_dotenv(Path.cwd() / ".env"), **_read_dotenv(base / ".env"), **os.environ}
        if env.get("OLLAMA_BASE_URL"):
            cfg.vision_url = env["OLLAMA_BASE_URL"]
        if env.get("OLLAMA_MODEL"):
            cfg.vision_model = env["OLLAMA_MODEL"]
        cfg.vision_api_key = env.get("OLLAMA_API_KEY", "")
        cfg.darwin_api_key = env.get("DARWIN_API_KEY", "")
        return cfg

    def save(self) -> None:
        base = Path(self.home)
        base.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k not in ("vision_api_key", "darwin_api_key")}
        (base / "config.json").write_text(json.dumps(data, indent=2))


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out
