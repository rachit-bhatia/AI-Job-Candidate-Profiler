import networkx as nx
from text_processing import chunk_text
from text_extraction import parse_pdf
from data import SourceDocument
from agents.extractor_agent import build_evidence_graph

def create_candidate_graph(candidate_repos: dict[str, dict[str, str]], cv_filepath: str) -> nx.DiGraph:
    candidate_source_docs = []

    #creating SourceDocument from every file in the candidate github repos 
    for repo_name, repo_contents in candidate_repos.items():
        for filename, file_content in repo_contents.items():
            file_chunks = chunk_text(file_content)
            candidate_source_docs.append(SourceDocument(
                source_id=f"{repo_name}-{filename}", 
                source_type="github_repos", 
                text=chunk) for chunk in file_chunks)

    #creating SourceDocument from the CV
    cv_text = parse_pdf(cv_filepath)
    cv_text_chunks = chunk_text(cv_text)
    candidate_source_docs.append(SourceDocument(source_id="Candidate CV", source_type="Candidate CV", text=chunk) for chunk in cv_text_chunks)

    #building evidence graph from candidate source documents
    candidate_graph = build_evidence_graph(candidate_source_docs)
    return candidate_graph