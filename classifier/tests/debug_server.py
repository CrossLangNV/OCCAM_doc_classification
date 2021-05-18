import uvicorn

from main import app

if __name__ == '__main__':
    print(app)
    uvicorn.run("debug_server:app",
                host="127.0.0.1",
                port=1234,
                reload=True,
                )
