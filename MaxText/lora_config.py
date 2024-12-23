from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 8 # LoRA rank
    alpha: int = 32 # LoRA scaling factor
    dropout: float = 0.0 # LoRA dropout
