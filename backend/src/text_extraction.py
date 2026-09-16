from pathlib import Path
from urllib.parse import urlparse
from unstructured.partition.pdf import partition_pdf
import requests

GITHUB_API_URL = "https://api.github.com"
SUPPORTED_EXTENSIONS = {".md", ".txt"}

def parse_github_url(profile_url):
    """Extract the GitHub username from a profile URL."""
    parsed_url = urlparse(profile_url)
    parts = parsed_url.path.strip("/").split("/")

    if (parsed_url.netloc.lower() != "github.com") or len(parts) < 1:
        raise ValueError("Given URL is not a GitHub profile URL")

    owner = parts[0]
    return owner

def fetch_files(owner):
    """Fetch supported root-level files from the user's repositories."""
    get_user_repos_api = f"{GITHUB_API_URL}/users/{owner}/repos"

    #fetching the repos owned by the user
    response = requests.get(get_user_repos_api,
                            params={
                                "type": "owner",
                                "sort": "updated", #sort by recently updated 
                                "per_page": 25,
                                "page": 1,
                            },
                            timeout=10)

    if response.status_code == 404:
        raise ValueError("GitHub profile not found")

    response.raise_for_status()
    user_repos = response.json()

    fetched_files = {}

    #fetching the md and txt files from each repo
    for repo in user_repos:
        get_repo_contents_api = f"{GITHUB_API_URL}/repos/{owner}/{repo['name']}/contents"
        response = requests.get(get_repo_contents_api, timeout=10)

        if response.status_code == 404:
            raise ValueError("GitHub repository not found")
    
        response.raise_for_status()
        directory_list = response.json()
        fetched_files[repo['name']] = []

        for item in directory_list:
            if item.get("type") != "file":
                continue

            filename = item.get("name", "")
            extension = Path(filename).suffix.lower()

            if extension in SUPPORTED_EXTENSIONS:
                fetched_files[repo['name']].append(item)

    return fetched_files

def fetch_file_content(file_object):
    """Fetch the text content of a GitHub file."""
    download_url = file_object.get("download_url")

    if not download_url:
        raise ValueError("File not downloadable from repository")
    
    response = requests.get(download_url)
    response.raise_for_status()
    return response.text

def get_contents(profile_url):
    """Fetch supported file contents from a GitHub profile."""
    username = parse_github_url(profile_url)
    fetched_files = fetch_files(username)

    repo_contents = {}

    for repo_name, files in fetched_files.items():
        repo_contents[repo_name] = {}
        for file in files:
            file_content = fetch_file_content(file)
            repo_contents[repo_name][file.get("name")] = file_content

    return repo_contents

def parse_pdf(file):
    """Extract text from a PDF file."""
    elements = partition_pdf(filename=file)
    pdf_content = ""

    for element in elements:
        if hasattr(element, "text"):
            text = element.text
            pdf_content += text + "\n\n"

    return pdf_content

print(get_contents("https://github.com/rachit-bhatia"))
        