import json
from pathlib import Path
from typing import Any, Dict, Union

from _jsonnet import evaluate_file

PROJECT_ROOT = str(Path(__file__).parent.parent.parent) + "/"


def load_jsonnet(path: Union[str, Path]) -> Dict[str, Any]:
    return json.loads(
        evaluate_file(
            str(path),
            ext_vars={"PROJECT_ROOT": PROJECT_ROOT},
        )
    )
