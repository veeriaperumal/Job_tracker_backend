from contextlib import asynccontextmanager

from fastapi import FastAPI
from core.response import JSONResponse

from core.exceptions import EXCEPTION_HANDLERS
from logs import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    yield


app = FastAPI(lifespan=lifespan,
              title= "JOB SEARCH AI")

for exc_type_or_code, handler in EXCEPTION_HANDLERS.items():
    app.add_exception_handler(exc_type_or_code, handler)

@app.get('/health')
def check_health():
    return JSONResponse(status_code=200,content={"message":"running success!!"})
