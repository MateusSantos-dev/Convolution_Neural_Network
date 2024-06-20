from fastapi import FastAPI
from API.router import network_router


app = FastAPI(
    title="Trabalho de Inteligência Artificial - CNN"
)

app.include_router(network_router)
