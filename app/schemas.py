from typing import Optional
from pydantic import BaseModel

class RunRequest(BaseModel):
    image_folder: Optional[str] = None
    llm_provider: Optional[str] = None
    api_key: Optional[str] = None
