from pydantic import BaseModel, Field

class TokenizeRequest(BaseModel):
    text: str = Field(description="Text to tokenize after training")
    