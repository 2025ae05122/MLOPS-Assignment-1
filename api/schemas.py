from pydantic import BaseModel, Field


class HeartDiseaseRequest(BaseModel):
    age: float = Field(..., ge=0, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., gt=0)
    chol: float = Field(..., ge=0)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., gt=0)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=3, le=7)


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str