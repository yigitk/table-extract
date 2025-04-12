import uuid as std_uuid
from datetime import date as date_type
from typing import Optional, Union

from pydantic import BaseModel, Field


class ProcessedBiomarker(BaseModel):
    """
    Represents a single processed biomarker extracted from a document.
    """
    uuid: std_uuid.UUID = Field(default_factory=std_uuid.uuid4, description="Unique identifier for this biomarker record.")
    biomarkerId: Optional[str] = Field(None, description="Identifier if matched against a known biomarker database.")
    rawBiomarkerName: str = Field(..., description="The name of the biomarker as extracted from the document.")
    matched: Optional[bool] = Field(None, description="Flag indicating if the biomarker was matched.")
    value: Optional[float] = Field(None, description="The numerical value of the biomarker result.")
    unit: Optional[str] = Field(None, description="The unit associated with the biomarker value.")
    labDate: Optional[date_type] = Field(None, description="The date extracted from the lab report (e.g., reported or collected date).")
    processingDate: date_type = Field(..., description="The date when the document was processed by the API.")

    class Config:
        # Example for documentation and schema generation
        json_schema_extra = {
            "example": {
                "uuid": "123e4567-e89b-12d3-a456-426614174000",
                "biomarkerId": "LOINC:1988-5",
                "rawBiomarkerName": "Glucose",
                "matched": True,
                "value": 105.5,
                "unit": "mg/dL",
                "labDate": "2023-10-25",
                "processingDate": "2023-10-27",
            }
        } 