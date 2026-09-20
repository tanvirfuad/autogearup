# Local CCTV Gateway

This gateway must run on a computer that can reach the NVR/camera over the local network.

## Why a gateway is required

Browsers cannot play RTSP directly, and a public GitHub Pages site cannot safely store camera credentials. The gateway keeps the RTSP URL private, converts the stream into a browser-readable MJPEG feed, and runs local AI analytics.

## Setup on Windows

1. Install Python 3.11 or 3.12.
2. Open PowerShell in this `gateway` folder.
3. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

4. Install dependencies:

```powershell
pip install -r requirements.txt
```

5. Copy `.env.example` to `.env` and put the real RTSP URL in `.env`.
6. Start the gateway:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

7. Open:

```text
http://localhost:8000/
```

From another device on the same LAN, open `http://<gateway-PC-IP>:8000/`.

## Analytics

The current gateway stores crossing events, plate reads and incidents in a local SQLite database (`cctv.db`). When the dashboard is opened through the gateway, Today / This Week / This Month / This Year / Custom use the stored real data.

The current gateway:
- tracks people
- tracks cars, motorcycles, buses and trucks
- counts tracked objects when they cross the configurable counting line
- records direction
- records vehicle-type totals
- can optionally run local OCR against likely plate regions
- can trigger a stopped-vehicle incident rule

The public GitHub Pages version remains a simulator. The live version is served by this gateway so the browser and camera stay on the same trusted network.

## Security

Do not put usernames, passwords, RTSP URLs containing credentials, or private network configuration into GitHub. Keep them in `.env` only.


## Multiple cameras

The dashboard now includes a Camera Connections screen.

1. Start the gateway with `start-windows.bat`.
2. Open `http://localhost:8000`.
3. Click **Cameras**.
4. Enter a camera name.
5. Paste the full RTSP URL exactly as you would in VLC.
6. Click **Add Camera**.

The gateway stores camera connections locally in `cctv.db` and starts them again after a restart. Camera credentials are never returned to the browser after saving; the UI only displays a masked connection string.

If an older installation already has `CAMERA_RTSP_URL` in `.env`, it is imported once as **Camera 1** automatically. Future cameras can be added from the dashboard without editing `.env`.

Removing a camera from the dashboard stops that worker and removes its saved connection. Historical event records remain in the local analytics database.
