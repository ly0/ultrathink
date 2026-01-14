#!/usr/bin/env python3
"""Verify ultrathink web API installation."""

import sys


def check_imports():
    """Check that all required modules can be imported."""
    errors = []

    # Check API models
    try:
        from ultrathink.api.models import Thread, Message, Assistant
        print("[OK] API models import successfully")
    except ImportError as e:
        errors.append(f"API models: {e}")

    # Check services
    try:
        from ultrathink.api.services import ThreadStore, AssistantRegistry, StreamManager
        print("[OK] API services import successfully")
    except ImportError as e:
        errors.append(f"API services: {e}")

    # Check routes
    try:
        from ultrathink.api.routes import health_router, threads_router, assistants_router, runs_router
        print("[OK] API routes import successfully")
    except ImportError as e:
        errors.append(f"API routes: {e}")

    # Check FastAPI app
    try:
        from ultrathink.api.app import create_app
        print("[OK] FastAPI app import successfully")
    except ImportError as e:
        errors.append(f"FastAPI app: {e}")

    return errors


def check_dependencies():
    """Check that required dependencies are installed."""
    errors = []

    deps = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("sse_starlette", "SSE Starlette"),
        ("aiofiles", "aiofiles"),
        ("pydantic", "Pydantic"),
    ]

    for module, name in deps:
        try:
            __import__(module)
            print(f"[OK] {name} installed")
        except ImportError:
            errors.append(f"{name} not installed (pip install {module})")

    return errors


def check_frontend():
    """Check that frontend files exist."""
    from pathlib import Path

    web_dir = Path(__file__).parent.parent / "ultrathink" / "web"

    required = [
        "package.json",
        "next.config.ts",
        "src/lib/config.ts",
    ]

    errors = []
    for file in required:
        path = web_dir / file
        if path.exists():
            print(f"[OK] Frontend file exists: {file}")
        else:
            errors.append(f"Missing frontend file: {file}")

    return errors


def main():
    """Run all checks."""
    print("=" * 50)
    print("Ultrathink Web API Verification")
    print("=" * 50)
    print()

    all_errors = []

    print("Checking dependencies...")
    all_errors.extend(check_dependencies())
    print()

    print("Checking imports...")
    all_errors.extend(check_imports())
    print()

    print("Checking frontend...")
    all_errors.extend(check_frontend())
    print()

    print("=" * 50)
    if all_errors:
        print("ERRORS FOUND:")
        for error in all_errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("All checks passed!")
        print()
        print("To start the web server:")
        print("  cd ultrathink")
        print("  pip install -e .")
        print("  ultrathink serve")
        print()
        print("To build the frontend:")
        print("  cd ultrathink/web")
        print("  yarn install")
        print("  yarn build")
        sys.exit(0)


if __name__ == "__main__":
    main()
