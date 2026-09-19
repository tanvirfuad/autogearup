# AutoGearUp CCTV Analytics Demo

Interactive static proof-of-concept for CCTV analytics.

Features:
- people and vehicle counting
- vehicle type classification
- readable license-plate events with confidence
- major incident monitoring
- real-time simulated detections
- historical period filters: today, week, month, year, custom
- responsive dashboard

This GitHub demo uses simulated data. Production architecture should keep raw RTSP streams local and use a local AI gateway for detection/tracking/OCR before sending event metadata and browser-safe preview streams to the web dashboard.

Open: `/cctv-analytics-demo/index.html`
