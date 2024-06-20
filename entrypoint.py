if __name__ == "__main__":
    import uvicorn

    # Inicia o servidor usando uvicorn
    uvicorn.run("API.main:app", host="0.0.0.0", port=6000, reload=False)
