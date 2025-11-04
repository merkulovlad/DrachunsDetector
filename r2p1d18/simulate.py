# save as ws_file_sender.py
import asyncio
import json
import time

import cv2
import websockets

WS_URL = "ws://localhost:8000/ws-frames"
VIDEO = "demo.mpeg"
FPS_SEND = 25  # throttle sender
STRIDE = 16


async def main():
    async with websockets.connect(WS_URL, max_size=None) as ws:
        await ws.send(json.dumps({"stride_frames": STRIDE}))
        cap = cv2.VideoCapture(VIDEO)
        if not cap.isOpened():
            print("Cannot open", VIDEO)
            return
        last = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # JPEG encode
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                continue
            now = time.time()
            if now - last < 1.0 / FPS_SEND:
                await asyncio.sleep((1.0 / FPS_SEND) - (now - last))
            last = time.time()
            await ws.send(buf.tobytes())
            # receive async without blocking send cadence
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.001)
                print("pred:", msg)
            except asyncio.TimeoutError:
                pass
        cap.release()


asyncio.run(main())
