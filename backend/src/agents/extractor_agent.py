import os
import uuid
from typing import List, Optional

import numpy as np
import networkx as nx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from text_processing import chunk_text
from data import SourceDocument, ExtractedClaim, SkillNode
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.5-flash-lite"
SKILL_MERGE_THRESHOLD = 0.80
embedder_model = SentenceTransformer("all-MiniLM-L6-v2")

EXTRACTION_FUNCTION_DECLARATION = {
            "name": "record_extracted_claims",
            "description": (
                "Record every distinct technical skill or capability the "
                "text provides direct evidence for. Only include skills you "
                "can point to a specific sentence for, do not infer skills "
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
job description, a candidate's resume, or code repository content for a hiring-screening tool.

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


#### Canonicalization Functions - merge near-duplicate skill mentions into one Skill node using embedding similarity.
class Canonicalizer:
    """
    Stateful per-candidate: holds one instance for the duration of one
    screening run, feeds it claims, and it incrementally builds up the set
    of distinct Skill nodes. 
    """
 
    def __init__(self, merge_threshold: float = SKILL_MERGE_THRESHOLD):
        self.merge_threshold = merge_threshold
        self.skill_nodes: List[SkillNode] = []
 
    def _find_matching_node(self, embedding: np.ndarray) -> Optional[SkillNode]:
        most_similar_node, best_matching_score = None, 0.0
        for node in self.skill_nodes:
            score = cosine_similarity(embedding.reshape(1, -1), node.embedding.reshape(1, -1))[0][0]
            if score > best_matching_score:
                most_similar_node, best_matching_score = node, score
        if most_similar_node and best_matching_score >= self.merge_threshold:
            return most_similar_node
        return None
 
    def canonicalize(self, claim: ExtractedClaim) -> str:
        """
        Assign a claim's skill phrasing to an existing Skill node, or create
        a new one. Returns the resulting skill_id.
        """
        embedding = embedder_model.encode(claim.skill, normalize_embeddings=True)
        match = self._find_matching_node(embedding)
        if match:
            match.member_labels.append(claim.skill)
            return match.skill_id
 
        new_node = SkillNode(
            skill_id=f"skill_{uuid.uuid4().hex[:8]}",
            canonical_label=claim.skill,
            embedding=embedding,
            member_labels=[claim.skill],
        )
        self.skill_nodes.append(new_node)
        return new_node.skill_id


# Evidence graph construction — wires extracted skills & canonicalization output into an in-memory NetworkX graph for this one candidate.
# Claim --EVIDENCED_BY--> SourceDocument
# Claim --DEMONSTRATES--> Skill
 
def build_evidence_graph(documents: List[SourceDocument]) -> nx.DiGraph:
    graph = nx.DiGraph()
    canonicalizer = Canonicalizer()
 
    for doc in documents:
        graph.add_node(doc.source_id, type="SourceDocument", source_type=doc.source_type)
 
    all_claims = extract_all_claims(documents)
 
    # pass 1: canonicalize every claim first, so every Skill node exists with its real attributes before any edge tries to reference it.
    claim_skill_pairs = [(claim, canonicalizer.canonicalize(claim)) for claim in all_claims]
 
    for node in canonicalizer.skill_nodes:
        graph.add_node(
            node.skill_id,
            type="Skill",
            canonical_label=node.canonical_label,
            member_labels=node.member_labels,
        )
 
    # pass 2: add claim nodes and wire up edges to nodes that already exist with full attributes
    for claim, skill_id in claim_skill_pairs:
        claim_id = f"claim_{uuid.uuid4().hex[:8]}"
        graph.add_node(
            claim_id,
            type="Claim",
            skill_label=claim.skill,
            evidence_sentence=claim.evidence_sentence,
            confidence=claim.confidence,
        )
        graph.add_edge(claim_id, claim.source_id, relation="EVIDENCED_BY")
        graph.add_edge(claim_id, skill_id, relation="DEMONSTRATES")
 
    return graph


if __name__ == "__main__":
    sample_docs = [
        SourceDocument(
            source_id="github_repo:order-processing-service",
            source_type="github_repo",
            text=(
                "Built with FastAPI and asyncio for concurrent order "
                "processing. Uses PostgreSQL for persistence with "
                "optimized queries using window functions for weekly "
                "analytics reports. Deployed via Docker Compose across "
                "staging and production environments."
            ),
        ),
        SourceDocument(
            source_id="resume",
            source_type="resume",
            text=(
                "Software Engineer with experience building backend "
                "services. Led migration of a monolith into containerized "
                "microservices, reducing deployment time by 60%."
            ),
        ),
    ]
 
    graph = build_evidence_graph(sample_docs)
 
    print("=== Skill nodes (after canonicalization) ===")
    for node_id, data in graph.nodes(data=True):
        if data["type"] == "Skill":
            print(f"- {data['canonical_label']}  (merged from: {data['member_labels']})")
 
    print("\n=== Claims and their edges ===")
    for node_id, data in graph.nodes(data=True):
        if data["type"] == "Claim":
            skill_id = [
                v for _, v, e in graph.out_edges(node_id, data="relation")
                if e == "DEMONSTRATES"
            ][0]
            source_id = [
                v for _, v, e in graph.out_edges(node_id, data="relation")
                if e == "EVIDENCED_BY"
            ][0]
            print(
                f"- Claim: {data['skill_label']!r} "
                f"(confidence: {data['confidence']})\n"
                f"    evidence: {data['evidence_sentence']!r}\n"
                f"    source: {source_id}\n"
                f"    -> Skill node: {graph.nodes[skill_id]['canonical_label']}"
            )