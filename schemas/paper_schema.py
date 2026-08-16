from pydantic import BaseModel, Field

class PaperMetadata(BaseModel):
    paper_id: str = Field(
        description="Unique identifier for the paper (e.g., arXiv ID)."
    )

    title: str = Field(
        description="Title of the paper."
    )

    authors: list[str] = Field(
        description="List of authors."
    )

    abstract: str = Field(
        description="Abstract of the paper."
    )

    pdf_url: str = Field(
        description="Direct URL to the PDF."
    )

    published: str = Field(
        description="Publication date."
    )

    source: str = Field(
        description="Source database (e.g. arXiv, Semantic Scholar)."
    )