import uvicorn

if __name__ == '__main__':
    uvicorn.run(
        "main:app",          # Points to main.py inside the app directory
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        app_dir="src"        # Explicitly tells uvicorn to look inside the "app" folder
    )
