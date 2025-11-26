# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from tasks import add  # Celery task

app = FastAPI()

class AddRequest(BaseModel):
    x: int
    y: int

@app.get("/")
def root():
    return {"message": "FastAPI + Celery demo"}

@app.post("/add")
def add_numbers(body: AddRequest):
    """
    Enqueue a Celery task and return the task_id.
    The client can use the task_id to check the result later.
    """
    # Call Celery asynchronously
    result = add.delay(body.x, body.y)

    return {
        "task_id": result.id,
        "status": "PENDING"
    }

@app.get("/result/{task_id}")
def get_result(task_id: str):
    """
    Check the status/result of a task.
    """
    from celery_app import celery_app

    async_result = celery_app.AsyncResult(task_id)

    if async_result.state == "PENDING":
        return {"task_id": task_id, "status": "PENDING"}

    if async_result.state == "STARTED":
        return {"task_id": task_id, "status": "STARTED"}

    if async_result.state == "FAILURE":
        # You can also return error info here
        return {
            "task_id": task_id,
            "status": "FAILURE",
            "error": str(async_result.info),
        }

    # SUCCESS (or other finished states)
    return {
        "task_id": task_id,
        "status": async_result.state,
        "result": async_result.result,
    }
