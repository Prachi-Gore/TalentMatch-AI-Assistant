from pathlib import Path
from typing import Any

import yaml
from langchain_core.output_parsers import PydanticOutputParser

from hireflow.config import settings
from hireflow.schemas import ResumeResponse


resume_parser = PydanticOutputParser(pydantic_object=ResumeResponse)


def load_prompt(path: Path | str = settings.prompt_path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


def render_resume_prompt(cfg: dict[str, Any], query: str, sources: Any) -> str:
    vars_all = dict(
        cfg.get("vars", {}),
        query=query,
        sources=sources,
        format_instructions=resume_parser.get_format_instructions(),
    )
    return cfg["template"].format(**vars_all)

