from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdown_it import MarkdownIt

md = MarkdownIt()

class DocumentElement:
    def __init__(self, text, element_type, metadata):
        self.text = text
        self.element_type = element_type
        self.metadata = metadata

def parse_pdf(file):
    elements = partition_pdf(filename=file)
    document_elements = []

    for element in elements:
        if hasattr(element, "text"):
            text = element.text
            element_type = type(element).__name__
            metadata = element.metadata.to_dict()
            document_elements.append(DocumentElement(text, element_type, metadata))

    return document_elements

def parse_text_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    return [DocumentElement(text, "Text", {})]

def parse_md_file(file):
    document_elements = []

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

    tokens = md.parse(text)
    para_id = 1
    i = 0

    while i < len(tokens): 
        if tokens[i].type == "heading_open":
            inline_token = tokens[i + 1]
            document_elements.append(
                DocumentElement(
                    inline_token.content,
                    "heading",
                    {"level": int(tokens[i].tag[1])}
                )
            )
            i += 3

        elif tokens[i].type == "paragraph_open":
            inline_token = tokens[i + 1]
            document_elements.append(
                DocumentElement(
                    inline_token.content,
                    "paragraph",
                    {"para_id": para_id}
                )
            )
            i += 3
            para_id += 1

        else:
            i += 1

    return document_elements

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

