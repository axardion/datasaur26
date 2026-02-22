import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "src_graph_rag"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src_graph_rag.local_embedding  # noqa: F401

rest = [a for a in sys.argv[1:] if a]
root_args = ["--root", str(CONFIG_ROOT)]
if "--root" not in rest:
    rest = root_args + rest
sys.argv = ["graphrag", "index"] + rest

from graphrag.cli.main import app

app()
