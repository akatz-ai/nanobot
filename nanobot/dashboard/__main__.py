"""Run the dashboard standalone: python -m nanobot.dashboard"""

import uvicorn


def main():
    uvicorn.run(
        "nanobot.dashboard.app:app",
        host="0.0.0.0",
        port=9347,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
