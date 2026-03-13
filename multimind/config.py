from dataclasses import dataclass
from pathlib import Path


APP_NAME = "Thinking-Wrapper"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
REQUEST_TIMEOUT_SECONDS = 90.0
DISCOVERY_TIMEOUT_SECONDS = 1.2
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.08
DEFAULT_PRESENCE_PENALTY = 0.1
DEFAULT_FREQUENCY_PENALTY = 0.25
REPETITION_SUFFIX_MIN_LENGTH = 32
REPETITION_SUFFIX_MAX_LENGTH = 220
REPETITION_SUFFIX_REPEAT_COUNT = 3

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"
PIPELINE_STEPS = ("plan", "execute", "critique")


@dataclass(frozen=True)
class ProviderCandidate:
    name: str
    kind: str
    base_url: str


DISCOVERY_CANDIDATES = (
    ProviderCandidate(
        name="Ollama",
        kind="ollama",
        base_url="http://127.0.0.1:11434",
    ),
    ProviderCandidate(
        name="LM Studio",
        kind="openai",
        base_url="http://127.0.0.1:1234",
    ),
)