from fastapi.responses import FileResponse
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stored_results = []

class EducationPredictor:
    def __init__(self):
        self.weights = [0.2, 0.3, 0.5]

    def weighted_average(self, values):
        y1, y2, y3 = values[-3:]
        return (self.weights[0] * y1 +
                self.weights[1] * y2 +
                self.weights[2] * y3)

    def apply_logic(self, value, indicator_type):
        if indicator_type == "positive":
            value += (100 - value) * 0.2
        elif indicator_type == "negative":
            value -= value * 0.2
        elif indicator_type == "gpi":
            target = 1.02
            value += (target - value) * 0.1
        return value

    def clamp(self, value, indicator_type):
        if indicator_type == "gpi":
            return max(0.95, min(1.05, value))
        else:
            return max(0, min(100, value))

    def predict(self, values, indicator_type):
        base = self.weighted_average(values)
        adjusted = self.apply_logic(base, indicator_type)
        final_value = self.clamp(adjusted, indicator_type)
        return round(final_value, 2)

model = EducationPredictor()

class InputData(BaseModel):
    retention: List[float]
    transition: List[float]
    promotion: List[float]
    dropout: List[float]
    repetition: List[float]
    gpi: List[float]

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/predict-all")
def predict_all(data: InputData):
    result = {
        "Retention Rate": model.predict(data.retention, "positive"),
        "Transition Rate": model.predict(data.transition, "positive"),
        "Promotion Rate": model.predict(data.promotion, "positive"),
        "Dropout Rate": model.predict(data.dropout, "negative"),
        "Repetition Rate": model.predict(data.repetition, "negative"),
        "Gender Parity Index": model.predict(data.gpi, "gpi")
    }

    stored_results.append({
        "input": data.dict(),
        "output": result
    })

    return result

@app.get("/results")
def get_results():
    return stored_results

@app.get("/latest")
def get_latest():
    if stored_results:
        return stored_results[-1]
    return {"message": "No data available"}
