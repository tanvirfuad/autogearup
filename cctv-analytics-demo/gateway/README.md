# Local CCTV Gateway — Four Fixed Cameras

This version uses the simpler dashboard with four fixed camera slots.

## Setup

1. Run the gateway on a Windows PC on the same local network as the NVR/cameras.
2. Copy `.env.example` to `.env`.
3. Put the four RTSP links into:
   - `CAMERA1_RTSP_URL`
   - `CAMERA2_RTSP_URL`
   - `CAMERA3_RTSP_URL`
   - `CAMERA4_RTSP_URL`
4. Optionally rename the four cameras with `CAMERA1_NAME` through `CAMERA4_NAME`.
5. Double-click `start-windows.bat`.
6. Open `http://localhost:8000`.

An older `CAMERA_RTSP_URL` setting is still accepted as Camera 1, so an existing working single-camera setup does not have to be changed immediately.

## Lorex-style example

The camera/channel number normally changes in the RTSP query:

```text
channel=1&subtype=1
channel=2&subtype=1
channel=3&subtype=1
channel=4&subtype=1
```

Use the same exact RTSP URLs that work in VLC.

## What the gateway does

Each configured camera gets its own live browser feed and AI worker. Events from all four cameras are combined in the dashboard and stored in the local SQLite database `cctv.db`.

The dashboard provides:
- four simultaneous live camera tiles
- people and vehicle crossing counts
- vehicle classification
- optional license-plate OCR
- stopped-vehicle incident monitoring
- combined historical filters for Today, This Week, This Month, This Year, and Custom Date

## Security

Keep the real camera usernames/passwords only in the local `.env` file. The `.env` file is excluded from GitHub.


## Secure public sharing

For a temporary externally shareable live dashboard:

1. Make sure the four camera URLs work locally.
2. Double-click `start-public-windows.bat`.
3. On first run it downloads Cloudflare's `cloudflared` client and generates a random 64-character access key in the local `.env`.
4. It starts:
   - the private CCTV gateway on port 8000
   - an authenticated read-only relay on port 8001
   - a Cloudflare Quick Tunnel to the relay only
5. The script opens the GitHub Pages dashboard with the tunnel URL and key in the URL fragment. The fragment is consumed by the browser and removed from the address bar.
6. Copy/share the resulting dashboard link if another authorized viewer needs access.

The relay exposes only camera health, historical analytics, and the four browser-safe MJPEG streams. It does not expose the RTSP URLs, `.env`, SQLite database file, or local filesystem.

The Quick Tunnel URL changes whenever the tunnel is restarted. For a stable production URL, replace the Quick Tunnel with a named Cloudflare Tunnel and a hostname you control.


## 16-30 camera optimization

The gateway now uses a **single shared batched detector** for all configured RTSP streams rather than loading a YOLO model per camera. It supports environment entries up through `CAMERA32_RTSP_URL`.

See `SCALING.md` for architecture, tuning, and the optional TensorRT path. AutoGearUp continues to store event metadata only; Lorex remains the video recorder.
