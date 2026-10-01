from sklearn.metrics.pairwise import cosine_similarity
from backend.src.data import SkillNode, SkillMatchResult, EntailmentResult, OverallAssessment, STRONG_MATCH, MODERATE_MATCH, WEAK_MATCH
from typing import Any, List
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
 
MODEL = "gemini-3.5-flash"

SKILL_MATCH_THRESHOLD = 0.80
VERIFIED_STATUS = "verified"
WEAK_STATUS = "weak"
MISSING_STATUS = "missing"
ENTAILMENT_PASS = "entailment"
ENTAILMENT_NEUTRAL = "neutral"
ENTAILMENT_CONTRADICTION = "contradiction"

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def match_candiate_skills(jd_requirements: list[SkillNode], candidate_skills: list[dict[str, Any]]):
    results = []

    for requirement in jd_requirements: 
        for skill in candidate_skills:
            similarity_score = cosine_similarity(skill["embedding"].reshape(1, -1), requirement.embedding.reshape(1, -1))[0][0]

            if similarity_score >= SKILL_MATCH_THRESHOLD:
                entailment = check_entailment(skill["member_labels"], requirement.canonical_label)

                if entailment.label == ENTAILMENT_PASS:
                    final_status = VERIFIED_STATUS
                elif entailment.label == ENTAILMENT_NEUTRAL:
                    final_status = WEAK_STATUS
                else:
                    final_status = MISSING_STATUS

                results.append(SkillMatchResult(
                    requirement=requirement.canonical_label,
                    final_status=final_status
                ))
    return results


ENTAILMENT_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="record_entailment_check",
            description=(
                "Classify whether the provided evidence (premises) genuinely "
                "support the job requirement (hypothesis)."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": [ENTAILMENT_PASS, ENTAILMENT_NEUTRAL, ENTAILMENT_CONTRADICTION],
                        "description": (
                            f"'{ENTAILMENT_PASS}' = the premises directly and "
                            "specifically prove the hypothesis is true. "
                            f"'{ENTAILMENT_NEUTRAL}' = related but do not actually confirm it. "
                            f"'{ENTAILMENT_CONTRADICTION}' = actively conflict with it."
                        ),
                    },
                    "justification": {
                        "type": "string",
                        "description": (
                            "One sentence explaining the label, referencing "
                            "specific wording from the premises."
                        ),
                    },
                },
                "required": ["label", "justification"],
            },
        )
    ]
)
 
ENTAILMENT_SYSTEM_PROMPT = """You are verifying whether a candidate's \
demonstrated skills genuinely meet a job requirement.
 
You will be given:
- PREMISES: all the ways the candidate showed evidence of a skill
- HYPOTHESIS: a specific job requirement
 
Be strict. Evaluate all the premises together. Only label f'{ENTAILMENT_PASS}' if \
a reasonable person would agree the premises collectively prove the \
hypothesis is clearly true. Similar-sounding technologies are NOT the same \
f'{ENTAILMENT_NEUTRAL}' rather than f'{ENTAILMENT_PASS}'.
f'{ENTAILMENT_CONTRADICTION}' rather than f'{ENTAILMENT_PASS}'."""
 
ENTAILMENT_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode=types.FunctionCallingConfigMode.ANY,
        allowed_function_names=["record_entailment_check"],
    )
)
 
 
def check_entailment(premises: List[str], hypothesis: str) -> EntailmentResult:
    """
    Run an entailment check against multiple evidence sentences at once.
    Uses Google's genai SDK to call Gemini with function calling.
    The model is forced to call the 'record_entailment_check' function with structured output.
    """
    premises_text = "\n".join([f"- {p}" for p in premises])
    prompt = f"PREMISES:\n{premises_text}\n\nHYPOTHESIS: {hypothesis}"
    
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=ENTAILMENT_SYSTEM_PROMPT,
            tools=[ENTAILMENT_TOOL],
            tool_config=ENTAILMENT_TOOL_CONFIG,
            max_output_tokens=300,
        ),
    )
    
    # Parse the response — look for a function_call in the first candidate's content
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.function_call and part.function_call.name == "record_entailment_check":
                    args = part.function_call.args or {}
                    return EntailmentResult(
                        label=args.get("label", ENTAILMENT_NEUTRAL),
                        justification=args.get("justification", ""),
                    )
    
    # Fallback if no function call was found or response was empty
    return EntailmentResult(
        label=ENTAILMENT_NEUTRAL, 
        justification="No entailment response parsed."
    )
 
def print_report(results: List[SkillMatchResult]) -> OverallAssessment:
    """
    Pretty-print the final report for a recruiter.
    
    Shows three states:
    - verified: the skill evidence logically supports the requirement
    - weak: related evidence exists but doesn't confirm the requirement
    - missing: no supporting evidence found in available sources
    """    
    print("\n" + "="*70)
    print("CANDIDATE SCREENING REPORT")
    print("="*70 + "\n")
    
    verified_count = sum(1 for r in results if r.final_status == VERIFIED_STATUS)
    weak_count = sum(1 for r in results if r.final_status == WEAK_STATUS)
    missing_count = sum(1 for r in results if r.final_status == MISSING_STATUS)
    
    print(f"Summary: {verified_count} verified | {weak_count} weak | {missing_count} missing\n")

    overall_assessment = ""
    
    if verified_count >= (0.8 * len(results)):
        overall_assessment = STRONG_MATCH
    elif verified_count + weak_count >= (0.8 * len(results)):
        overall_assessment = MODERATE_MATCH
    else:
        overall_assessment = WEAK_MATCH

    return OverallAssessment(
        status=overall_assessment,
        verified_req_count=verified_count,
        weak_req_count=weak_count,
        missing_req_count=missing_count
    )