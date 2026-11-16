# WebSpark AR Cube — experimental baseline

A clean, free prototype using Three.js and jsQR. No MindAR, A-Frame, avatar or paid services.

## Deploy
Upload `index.html`, `main.js`, `marker.html` to the root of your GitHub repository, then enable GitHub Pages (Settings > Pages > Deploy from branch > main / root). Open the HTTPS Pages URL on your phone. Click **Print QR marker** and print at **100%** scale, then open the main page on the phone and press **Start camera**.

## Important limitations
- This is an initial diagnostic prototype, **not** the final stable tracker.
- QR is decoded every ~65ms; no optical flow is implemented yet.
- Homography uses an approximate camera intrinsic matrix; perspective alignment and distance can be inaccurate.
- A 60mm printed QR is assumed. Printing at another size invalidates scale.
- The renderer currently uses a simple 60° FOV and `object-fit:cover`; phone-specific camera calibration and video crop alignment are needed for precise AR registration.
- The marker QR opens the current deployed GitHub Pages index URL. `marker.html` must be opened at the deployed site, not as a local file.
- If detection fails, improve lighting, reduce motion blur and ensure the entire QR is visible.

Next: calibrate video/renderer projection, add corner tracking between QR detections, improve pose estimation and measure jitter/latency before replacing the cube with an avatar.
