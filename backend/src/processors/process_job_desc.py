from text_processing import chunk_text
from data import SourceDocument, SkillNode
from agents.extractor_agent import extract_all_claims, Canonicalizer

def extract_skills_from_jd(jd_text: str) -> list[SkillNode]:
    jd_chunks = chunk_text(jd_text)
    jd_docs = [SourceDocument(source_id="job_description", source_type="job_description", text=chunk) for chunk in jd_chunks]
    jd_claims = extract_all_claims(jd_docs)

    # canonicalize all extracted claims to unify skill representations of the job description
    canonicalizer = Canonicalizer()
    for claim in jd_claims:
        canonicalizer.canonicalize(claim)

    return canonicalizer.skill_nodes