cat > hauif_system/core/models.py << 'EOL'
from pydantic import BaseModel, Field
from typing import List, Optional

class Document(BaseModel):
    type: str
    content: str

class CorruptionCase(BaseModel):
    case_id: str = Field(..., description="Unique case identifier")
    claim: str = Field(..., description="Description of the corruption claim")
    documents: List[Document] = Field(..., description="List of relevant documents")
    defendants: List[str] = Field(..., description="List of defendants")
    key_issues: List[str] = Field(..., description="List of key issues")
    date: Optional[str] = Field(None, description="Optional date of the case")
EOL
