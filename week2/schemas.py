from typing import Literal

from pydantic import BaseModel, EmailStr, Field


class SupportTicket(BaseModel):
    customer_name: str = Field(description="Full name of the customer, or 'Unknown' if not stated in the email")
    customer_email: EmailStr = Field(description="The customer's email address exactly as written in the message")
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Urgency of the issue, judged from tone and content"
    )
    issue_type: Literal["billing", "technical", "account", "feature_request", "bug", "other"] = Field(
        description="Category that best describes the issue"
    )
    summary: str = Field(description="A one to two sentence summary of the customer's issue")
