"""
Run GraphRAG index using the local embedder (no external API calls).
Usage: uv run python -m src.run_index_with_local_embedding -v
Pass through any graphrag index args after the module name.
"""
import sys
from pathlib import Path

# Repo root; config (settings.yaml) is in src/
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "src"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register local embedder before graphrag CLI runs
import src.local_embedding  # noqa: F401

# Build argv: graphrag index --root <config_dir> [user args...]
# So GraphRAG finds settings.yaml in src/
rest = [a for a in sys.argv[1:] if a]
root_args = ["--root", str(CONFIG_ROOT)]
if "--root" not in rest:
    rest = root_args + rest
sys.argv = ["graphrag", "index"] + rest

from graphrag.cli.main import app

app()
