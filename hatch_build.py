"""Custom hatch build hook to build the frontend."""

import subprocess
import shutil
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class FrontendBuildHook(BuildHookInterface):
    """Build the Next.js frontend during package build."""

    PLUGIN_NAME = "frontend-build"

    def initialize(self, version: str, build_data: dict) -> None:
        """Run frontend build before packaging."""
        root = Path(self.root)
        web_dir = root / "ultrathink" / "web"
        out_dir = web_dir / "out"

        # Skip if configured
        if self.config.get("skip_frontend", False):
            self.app.display_info("Skipping frontend build (skip_frontend=true)")
            return

        # Check for yarn or npm
        yarn_path = shutil.which("yarn")
        npm_path = shutil.which("npm")

        if not yarn_path and not npm_path:
            self.app.display_warning(
                "Neither yarn nor npm found. Skipping frontend build. "
                "Run 'scripts/build_frontend.sh' manually."
            )
            return

        pkg_manager = "yarn" if yarn_path else "npm"

        if not (web_dir / "package.json").exists():
            self.app.display_warning(f"No package.json in {web_dir}")
            return

        self.app.display_info(f"Building frontend with {pkg_manager}...")

        try:
            # Install dependencies
            self.app.display_info("Installing frontend dependencies...")
            subprocess.run(
                [pkg_manager, "install"],
                cwd=web_dir,
                check=True,
                capture_output=True,
            )

            # Build
            self.app.display_info("Building frontend for production...")
            subprocess.run(
                [pkg_manager, "run", "build"],
                cwd=web_dir,
                check=True,
                capture_output=True,
            )

            if out_dir.exists():
                self.app.display_success("Frontend built successfully!")
            else:
                self.app.display_warning("Build completed but output not found")

        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else ""
            self.app.display_warning(
                f"Frontend build failed: {stderr[:200]}. "
                "Run 'scripts/build_frontend.sh' manually."
            )
        except Exception as e:
            self.app.display_warning(f"Frontend build error: {e}")
