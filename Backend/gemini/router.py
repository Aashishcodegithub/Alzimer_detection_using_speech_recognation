from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from auth import get_current_user, User
from .alz_chat import generate_alz_answer, is_alzheimer_question

router = APIRouter(prefix="/gemini", tags=["gemini"])


class AlzChatRequest(BaseModel):
    question: str
    class_name: Optional[str] = None
    probability: Optional[float] = None
    model_name: Optional[str] = None


class AlzChatResponse(BaseModel):
    answer: str
    rejected: bool = False


@router.post("/alz-chat", response_model=AlzChatResponse)
async def alz_chat_endpoint(
    req: AlzChatRequest,
    current_user: User = Depends(get_current_user),
):

    if not is_alzheimer_question(req.question):
        return AlzChatResponse(
            answer=(
                "I’m designed to answer questions related to Alzheimer’s disease, "
                "dementia, memory loss, and brain health. Please ask an Alzheimer-related question."
            ),
            rejected=True,
        )

    try:
        answer = generate_alz_answer(
            question=req.question,
            class_name=req.class_name,
            probability=req.probability,
            model_name=req.model_name,
        )
        return AlzChatResponse(answer=answer, rejected=False)

    except Exception as e:
        print("Gemini exception:", e)
        return AlzChatResponse(
            answer=(
                "An educational explanation is temporarily unavailable. "
                "Please consult a healthcare professional for medical guidance."
            ),
            rejected=False,
        )
