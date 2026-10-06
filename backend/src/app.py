from fastapi import FastAPI, HTTPException
from text_extraction import get_contents
from processors.process_candidate_docs import create_candidate_graph
from processors.process_job_desc import extract_skills_from_jd
from agents.matcher_agent import match_candiate_skills, print_report

app = FastAPI()

@app.get("/github-contents")
def get_github_contents(github_url: str):
    try:
        contents = get_contents(github_url)
        return {"valid": True, "contents": contents}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/match-candidate")
def match_candidate(candidate_repos: dict[str, dict[str, str]], cv_filepath: str, jd_text: str):
    candidate_skill_graph = create_candidate_graph(candidate_repos, cv_filepath)
    candidate_skill_nodes = [
        data
        for _, data in candidate_skill_graph.nodes(data=True)
        if data.get("type") == "Skill"
    ]
    jd_skills = extract_skills_from_jd(jd_text)

    #match the JD skills against the candidate skills
    match_results = match_candiate_skills(jd_skills, candidate_skill_nodes)
    match_report = print_report(match_results)

    return {"result": match_report}

    