#Run API bang uvicorn
import uvicorn
from fast_api import app #Import Fast_api module vao Server.
#import fast_api
if __name__ == "__main__":
    uvicorn.run("fast_api:app", host="0.0.0.0", port=8080, reload=True)