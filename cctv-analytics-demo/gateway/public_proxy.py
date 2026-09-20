import os, secrets

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

load_dotenv()

LOCAL_GATEWAY = os.getenv('LOCAL_GATEWAY_URL', 'http://127.0.0.1:8000').rstrip('/')
ACCESS_KEY = os.getenv('REMOTE_ACCESS_KEY', '')
ALLOWED_ORIGIN = os.getenv('REMOTE_ALLOWED_ORIGIN', 'https://tanvirfuad.github.io').rstrip('/')

app = FastAPI(title='AutoGearUp CCTV Public Relay')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_credentials=False,
    allow_methods=['GET', 'OPTIONS'],
    allow_headers=['*'],
)

def authorize(key: str):
    if not ACCESS_KEY:
        raise HTTPException(status_code=503, detail='Remote access key is not configured')
    if not secrets.compare_digest(key or '', ACCESS_KEY):
        raise HTTPException(status_code=401, detail='Invalid access key')

@app.get('/health')
async def health(key: str = Query(...)):
    authorize(key)
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f'{LOCAL_GATEWAY}/api/health')
        r.raise_for_status()
        return r.json()

@app.get('/history')
async def history(key: str = Query(...), period: str = 'today', start: str | None = None, end: str | None = None):
    authorize(key)
    params = {'period': period}
    if start:
        params['start'] = start
    if end:
        params['end'] = end
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(f'{LOCAL_GATEWAY}/api/history', params=params)
        r.raise_for_status()
        return r.json()

@app.get('/cameras/{camera_id}/stream')
async def camera_stream(camera_id: int, key: str = Query(...)):
    authorize(key)
    async def relay():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream('GET', f'{LOCAL_GATEWAY}/api/cameras/{camera_id}/stream') as response:
                if response.status_code != 200:
                    return
                async for chunk in response.aiter_raw():
                    if chunk:
                        yield chunk
    return StreamingResponse(
        relay(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate',
            'X-Content-Type-Options': 'nosniff',
            'Referrer-Policy': 'no-referrer',
        },
    )
