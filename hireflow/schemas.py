from typing import Optional

from pydantic import BaseModel, Field


class ResumeItem(BaseModel):
    resume_id: str = Field(..., description="Unique resume id.")
    filename: Optional[str] = Field(None, description="Original resume file name.")
    link: Optional[str] = Field(None, description="Path or URL to the resume.")
    jd_relevance: float = Field(..., description="Relevance score between 0 and 1.")
    profile_summary: str = Field(..., description="Candidate summary for the query.")
    key_skills: list[str] = Field(default_factory=list)
    risks_or_flags: list[str] = Field(default_factory=list)


class ResumeResponse(BaseModel):
    query: str
    resumes: list[ResumeItem]

