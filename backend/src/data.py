from dataclasses import dataclass, field
import numpy as np
from typing import List

@dataclass
class SourceDocument:
    source_id: str          # e.g. "github_repo:order-processing-service"
    source_type: str        # "resume" | "github_repo"
    text: str

@dataclass
class ExtractedClaim:
    skill: str               # model's own phrasing, e.g. "asyncio for concurrent I/O"
    evidence_sentence: str    # the exact sentence/snippet that supports it
    source_id: str
    confidence: str           # "high" | "medium" | "low" (self-reported by the model)

@dataclass
class SkillNode:
    skill_id: str
    canonical_label: str            # label of the first mention that created this node
    embedding: np.ndarray
    member_labels: List[str] = field(default_factory=list)  # every raw phrasing merged in