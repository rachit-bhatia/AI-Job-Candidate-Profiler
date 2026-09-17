import os
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv

from google import genai
from google.genai import types
from text_processing import chunk_text

load_dotenv()
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

MODEL = "gemini-3.5-flash"

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

EXTRACTION_FUNCTION_DECLARATION = {
            "name": "record_extracted_claims",
            "description": (
                "Record every distinct technical skill or capability the "
                "text provides direct evidence for. Only include skills you "
                "can point to a specific sentence for — do not infer skills "
                "the text doesn't actually demonstrate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "skill": {
                                    "type": "string",
                                    "description": (
                                        "The skill/capability in your own "
                                        "words, e.g. 'asyncio for concurrent "
                                        "I/O in FastAPI'. Be specific, not "
                                        "generic — prefer 'optimized "
                                        "PostgreSQL queries with window "
                                        "functions' over just 'SQL'."
                                    ),
                                },
                                "evidence_sentence": {
                                    "type": "string",
                                    "description": (
                                        "The exact sentence or snippet from "
                                        "the source text that supports this "
                                        "claim. Quote it verbatim, don't "
                                        "paraphrase."
                                    ),
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                    "description": (
                                        "'high' = explicit, unambiguous "
                                        "mention. 'medium' = clearly implied "
                                        "but not named directly. 'low' = "
                                        "plausible but a stretch."
                                    ),
                                },
                            },
                            "required": ["skill", "evidence_sentence", "confidence"],
                        },
                    }
                },
                "required": ["claims"],
            },
}

SYSTEM_PROMPT = """You are extracting evidence of technical skills from a \
candidate's resume or code repository content for a hiring-screening tool.

Rules:
- Only extract claims you can point to a specific sentence for. Never infer \
a skill that isn't actually stated or clearly demonstrated in the text.
- Do not round up. If someone mentions "familiar with Docker," don't record \
"production Kubernetes experience."
- Prefer specific, concrete phrasing over generic category names.
- If the text contains no clear technical skill evidence, call the tool \
with an empty claims list — do not force a match.
- Every claim must be traceable to an exact quoted sentence, not a summary."""

EXTRACTION_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(**EXTRACTION_FUNCTION_DECLARATION)
    ]
)

FORCED_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode=types.FunctionCallingConfigMode.ANY,
        allowed_function_names=["record_extracted_claims"],
    )
)

def extract_claims_from_chunk(chunk: str, source_id: str) -> List[ExtractedClaim]:
    response = client.models.generate_content(
        model=MODEL,
        contents=f"Source: {source_id}\n\nText:\n{chunk}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_config=FORCED_TOOL_CONFIG,
            max_output_tokens=1500,
        ),
    )

    #finding the function_call part and pulling its structured args directly
    #no string parsing, no risk of malformed JSON.
    candidates = response.candidates or []
    if not candidates or candidates[0].content is None:
        return []

    for part in candidates[0].content.parts or []:
        if part.function_call and part.function_call.name == "record_extracted_claims":
            args = part.function_call.args or {}
            raw_claims = args.get("claims", [])
            extracted_claims = []

            for claim in raw_claims:
                extracted_claims.append(
                    ExtractedClaim(
                        skill=claim["skill"],
                        evidence_sentence=claim["evidence_sentence"],
                        source_id=source_id,
                        confidence=claim["confidence"],
                    )
                )
            return extracted_claims
    return []


def extract_claims_from_document(doc: SourceDocument) -> List[ExtractedClaim]:
    all_claims: List[ExtractedClaim] = []
    for chunk in chunk_text(doc.text):
        all_claims.extend(extract_claims_from_chunk(chunk, doc.source_id))
    return all_claims


def extract_all_claims(documents: List[SourceDocument]) -> List[ExtractedClaim]:
    """Entry point: run extraction across every source document for one candidate."""
    all_claims: List[ExtractedClaim] = []
    for doc in documents:
        all_claims.extend(extract_claims_from_document(doc))
    return all_claims