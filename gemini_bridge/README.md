# Google Flow Image Bridge

This localhost-only bridge lets FAA generate YouTube thumbnails through the
real Google Flow web application in a headed Chrome session. FAA keeps using
the OpenAI-compatible `POST /v1/images/generations` endpoint on
`127.0.0.1:4981`, while the bridge delegates browser automation to the pinned
MIT-licensed `gflow-cli` driver.

The Google login is stored outside the repository in a dedicated persistent
Chrome profile. Chrome owns the cookie jar and refreshes an ordinary valid
Google session automatically. No Google cookies, password, or 2FA secret are
stored in `.env`.

## One-time setup on the production PC

From the FAA repository root:

```powershell
powershell -ExecutionPolicy Bypass -File ".\gemini_bridge\setup_browser_profile.ps1"
```

The script:

1. Creates `gemini_bridge\.env` with a random local bridge key when missing.
2. Installs the pinned Flow driver into the FAA Python environment.
3. Opens a dedicated real Chrome profile.
4. Waits while you sign in to Google and verifies the Flow session without
   spending credits.

Do not open that dedicated profile yourself while FAA is generating. Each Flow
request opens a headed Chrome window with the same profile and closes it after
the image is downloaded. The normal personal Chrome profile can stay open.

## FAA settings

Enable **Google Flow Image Bridge** and use:

- Bridge URL: `http://127.0.0.1:4981`
- Local Bridge API Key: the `LOCAL_API_KEY` value from `gemini_bridge\.env`
- Image Model: `flow-nano-pro`
- Timeout: `600`

Flow is forced to one output at `16:9`. FAA then normalizes the returned image
to exact `1920x1080` and saves it as `thumbnail_generated.png`.

## Start

Start Flow bridge and FAA together:

```powershell
powershell -ExecutionPolicy Bypass -File ".\start_faa_with_gemini.ps1"
```

## Session recovery

Normal cookie expiry is handled by Chrome's persistent profile. The bridge
also retries transient Flow/UI failures. If Google fully signs the account out
or requires password/2FA again, unattended recovery is intentionally not
attempted. Re-run the one-time setup command and sign in in the opened window.

Useful checks:

```powershell
Invoke-RestMethod "http://127.0.0.1:4981/health"
```

Authenticated session status is available at `/auth/status` with the same
local Bearer key used by FAA.
