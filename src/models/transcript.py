from dataclasses import dataclass
from pydantic import BaseModel


class Word(BaseModel):
    text: str
    start: int
    end: int


class Transcript(BaseModel):
    text: str
    words: list[Word]


class Sentence(BaseModel):
    text: str
    start: int
    end: int


class Hook(BaseModel):
    sentences: list[Sentence]