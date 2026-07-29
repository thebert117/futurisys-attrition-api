from fastapi import FastAPI

app = FastAPI(title="Futurisys Attrition API")

@app.get("/health")
def health():
    return {"status": "ok"}