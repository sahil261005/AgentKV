"""Request and response models for the API (Pydantic v2)."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


class Edge(BaseModel):
    src: str
    dst: str
    probability: float = Field(1.0, gt=0, le=1)


class WorkflowRequest(BaseModel):
    edges: list[Edge] = Field(min_length=1)
    num_blocks: int = Field(300, ge=1, le=100_000)
    # Belady is not offered here: it needs to know the future, which a live server can't.
    policy: Literal["LRU", "LRU-TailFirst", "AgentKV", "AgentKV-Markov"] = "AgentKV"
    static_tokens: int | None = Field(None, ge=0, description="length of each agent's fixed prompt")

    @model_validator(mode="after")
    def probabilities_add_up(self):
        totals = {}
        for edge in self.edges:
            totals[edge.src] = totals.get(edge.src, 0) + edge.probability
        for src, total in totals.items():
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"edge probabilities out of '{src}' add up to {total}, expected 1")
        return self


class WorkflowResponse(BaseModel):
    agents: list[str]
    num_blocks: int
    memory_mib: float
    policy: str


class StepRequest(BaseModel):
    agent: str
    tokens: list[int] = Field(min_length=1, max_length=200_000)


class StepResponse(BaseModel):
    agent: str
    prompt_tokens: int
    hit_tokens: int
    computed_tokens: int
    recomputed_tokens: int
    evicted_blocks: int
    free_blocks: int


class StatsResponse(BaseModel):
    policy: str
    capacity_blocks: int
    used_blocks: int
    free_blocks: int
    used_mib: float
    requests: int
    hit_rate: float
    hit_blocks: int
    total_blocks: int
    computed_tokens: int
    recomputed_tokens: int
    evictions: int
    evicted_blocks: int


class BenchmarkRequest(BaseModel):
    workflow: Literal["cyclic", "branching", "react", "fanout", "random"] = "branching"
    capacities: list[Annotated[int, Field(ge=16, le=5000)]] = Field(default=[240, 300, 360], min_length=1, max_length=10)
    seeds: int = Field(3, ge=1, le=10)
    n_steps: int = Field(200, ge=10, le=1000)


class BenchmarkRow(BaseModel):
    capacity_blocks: int
    capacity_mib: float
    hit_rate: dict[str, float]
    recomputed_tokens: dict[str, float]


class BenchmarkResponse(BaseModel):
    workflow: str
    seeds: int
    rows: list[BenchmarkRow]
