import json
from langchain_core.documents import Document

with open("C:\\Users\\paogr\\Desktop\\Projeto-NLP\\data\\processed\\extracted_json\\1-08 - Ressarcimento_sem_realce_2014.1.10_(jan-14).json", encoding = "utf-8") as arquivo:
    documento = json.load(arquivo)
    for block in documento["content_blocks"]:
        metadata = {
            **documento["metadata"],              # file_name, num_pages, author, etc.
            "file_path": documento["file_path"],
            "page": block["page"],
            "block_type": block["type"],
            "block_source": block["source"],
        }

        documento1 = Document(
            page_content=block["content"],
            metadata=metadata)