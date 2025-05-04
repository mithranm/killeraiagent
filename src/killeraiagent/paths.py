from pathlib import Path
from pydantic import BaseModel, Field, validator

class DataPaths(BaseModel):
    """Paths for application data storage."""
    base: Path
    models: Path = Field()
    logs: Path = Field()
    config: Path = Field()
    temp: Path = Field()
    grammars: Path = Field()
    templates: Path = Field()
    docs: Path = Field()
    bin: Path = Field()
    
    @validator("models", "logs", "config", "bin", pre=True, always=True)
    def set_derived_paths(cls, v, values):
        """Set derived paths if not explicitly provided."""
        if v is None and "base" in values:
            return values["base"] / "models"
        return v
    
    def ensure_dirs_exist(self):
        """Create all directories if they don't exist."""
        for path in [self.base, self.models, self.logs, self.config, self.bin]:
            path.mkdir(parents=True, exist_ok=True)
        return self

def get_data_paths() -> DataPaths:
    """
    Returns DataPaths configured to save data in the user's home directory.
    This fixes the issue where the base directory was set to a literal "~".
    """
    base = Path.home() / ".kaia"
    dp = DataPaths(base=base, models=base / "models",
                   logs=base / "logs", config=base / "config", bin=base / "bin",
                   temp=base / "temp", grammars=base / "grammars", templates=base / "templates",
                   docs=base / "docs")
    dp.ensure_dirs_exist()
    return dp