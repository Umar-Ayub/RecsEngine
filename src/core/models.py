from pydantic import BaseModel
from typing import List, Dict, Any

class Post(BaseModel):
    post_id: int
    text: str
    reported_or_removed: bool = False

class UserConv(BaseModel):
    ref_user_id: int
    messages_list: List[Dict[str, Any]]
