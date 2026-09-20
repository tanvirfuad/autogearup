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
