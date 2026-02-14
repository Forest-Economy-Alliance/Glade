from pydantic import BaseModel

class RunRequest(BaseModel):
    image_folder: str
    llm_provider: str
    api_key: str
