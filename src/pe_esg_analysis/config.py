from dataclasses import dataclass, field


@dataclass(frozen=True)
class PipelineConfig:
    min_sentence_length: int = 20
    climate_threshold: float = 0.35
    specificity_threshold: float = 0.40
    relevance_thresholds: tuple[float, float] = (2.0, 5.0)
    climate_keywords: tuple[str, ...] = field(default_factory=lambda: (
        "climate", "carbon", "emission", "greenhouse gas", "ghg",
        "net zero", "decarbon", "flood", "drought", "wildfire",
        "extreme weather", "sea level", "hurricane", "tornado", "blizzard",
        "storm", "stranded asset", "transition risk",
    ))
