# Gemini Web Image Bridge

This is an optional local bridge for generating thumbnail images through the
Gemini Web session. It binds to `127.0.0.1` and requires a separate local API
key. Browser mode uses a separate persistent Chrome profile, so cookies do not
need to be copied into the repository or refreshed manually before every image.

## Setup

1. Copy `.env.example` to `.env`.
2. Set a long random `LOCAL_API_KEY`.
3. Run the one-time browser setup from the repository root:

   ```powershell
   powershell -ExecutionPolicy Bypass -File ".\gemini_bridge\setup_browser_profile.ps1"
   ```

   A separate **normal** Chrome window opens. Sign in to Gemini there, close
   that dedicated window, and then press Enter in the PowerShell window. The
   profile is stored outside the repository under
   `%LOCALAPPDATA%\FAA\gemini_browser_profile`.
4. Put the same `LOCAL_API_KEY` into FAA Settings under **Gemini Web Image
   Bridge**, enable image generation, and save.

The old `GEMINI_1PSID` and `GEMINI_1PSIDTS` cookie fields remain only as a
fallback. With `GEMINI_BROWSER_MODE=auto`, an existing browser profile takes
priority. If Google signs the profile out, sign in again in the same bridge
window; no cookie copying is needed.

Start the bridge and FAA together from the repository root with:

```powershell
.\start_faa_with_gemini.ps1
```

The bridge exposes only the local `/health`, `/v1/models`, and
`/v1/images/generations` endpoints. Generated images are stored in each FAA
project as `thumbnail_generated.png` and are included by the downloader as
`thumbnail.png`.
