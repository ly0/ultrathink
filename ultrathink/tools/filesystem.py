"""Filesystem tools for Ultrathink.

These tools are used when not using deepagents' FilesystemMiddleware
(e.g., when using DeepSeek Reasoner).
"""

import os
import glob as glob_module
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import tool


def create_filesystem_tools(cwd: Optional[Path] = None) -> List:
    """Create filesystem tools.

    Args:
        cwd: Current working directory

    Returns:
        List of filesystem tools
    """
    working_dir = cwd or Path.cwd()

    @tool
    def ls(path: str = ".") -> str:
        """List files and directories in a path.

        Args:
            path: Directory path to list (default: current directory)

        Returns:
            List of files and directories
        """
        try:
            target = Path(path)
            if not target.is_absolute():
                target = working_dir / path

            if not target.exists():
                return f"Error: Path does not exist: {target}"

            if not target.is_dir():
                return f"Error: Not a directory: {target}"

            items = []
            for item in sorted(target.iterdir()):
                if item.is_dir():
                    items.append(f"{item.name}/")
                else:
                    items.append(item.name)

            return "\n".join(items) if items else "(empty directory)"

        except Exception as e:
            return f"Error: {e}"

    @tool
    def read_file(file_path: str, offset: int = 0, limit: int = 500) -> str:
        """Read a file's contents.

        Args:
            file_path: Path to the file to read
            offset: Line number to start reading from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            File contents with line numbers
        """
        try:
            target = Path(file_path)
            if not target.is_absolute():
                target = working_dir / file_path

            if not target.exists():
                return f"Error: File does not exist: {target}"

            if not target.is_file():
                return f"Error: Not a file: {target}"

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Apply offset and limit
            selected_lines = lines[offset : offset + limit]

            # Format with line numbers
            result = []
            for i, line in enumerate(selected_lines, start=offset + 1):
                result.append(f"{i:6d}\t{line.rstrip()}")

            return "\n".join(result)

        except Exception as e:
            return f"Error: {e}"

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write content to a file.

        Args:
            file_path: Path to the file to write
            content: Content to write

        Returns:
            Success message or error
        """
        try:
            target = Path(file_path)
            if not target.is_absolute():
                target = working_dir / file_path

            # Create parent directories if needed
            target.parent.mkdir(parents=True, exist_ok=True)

            with open(target, "w", encoding="utf-8") as f:
                f.write(content)

            return f"Successfully wrote to {target}"

        except Exception as e:
            return f"Error: {e}"

    @tool
    def edit_file(file_path: str, old_string: str, new_string: str) -> str:
        """Edit a file by replacing text.

        Args:
            file_path: Path to the file to edit
            old_string: Text to find and replace
            new_string: Replacement text

        Returns:
            Success message or error
        """
        try:
            target = Path(file_path)
            if not target.is_absolute():
                target = working_dir / file_path

            if not target.exists():
                return f"Error: File does not exist: {target}"

            with open(target, "r", encoding="utf-8") as f:
                content = f.read()

            if old_string not in content:
                return f"Error: Could not find the specified text in {target}"

            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                return f"Error: Found {count} occurrences of the text. Please provide more context to make it unique."

            new_content = content.replace(old_string, new_string)

            with open(target, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"Successfully edited {target}"

        except Exception as e:
            return f"Error: {e}"

    @tool
    def glob(pattern: str, path: str = ".") -> str:
        """Find files matching a glob pattern.

        Args:
            pattern: Glob pattern (e.g., '**/*.py')
            path: Base directory to search in

        Returns:
            List of matching file paths
        """
        try:
            target = Path(path)
            if not target.is_absolute():
                target = working_dir / path

            matches = list(target.glob(pattern))
            matches = [str(m) for m in matches[:100]]  # Limit results

            return "\n".join(matches) if matches else "No matches found"

        except Exception as e:
            return f"Error: {e}"

    @tool
    def grep(pattern: str, path: str = ".", glob_pattern: str = "*") -> str:
        """Search for a pattern in files.

        Args:
            pattern: Text pattern to search for
            path: Directory to search in
            glob_pattern: Glob pattern to filter files (e.g., '*.py')

        Returns:
            Matching lines with file paths
        """
        try:
            target = Path(path)
            if not target.is_absolute():
                target = working_dir / path

            results = []
            files = list(target.rglob(glob_pattern))[:100]  # Limit files

            for file_path in files:
                if not file_path.is_file():
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                results.append(f"{file_path}:{i}: {line.rstrip()}")
                                if len(results) >= 50:  # Limit results
                                    break
                except Exception:
                    continue

                if len(results) >= 50:
                    break

            return "\n".join(results) if results else "No matches found"

        except Exception as e:
            return f"Error: {e}"

    @tool
    def execute(command: str) -> str:
        """Execute a shell command.

        Args:
            command: The command to execute

        Returns:
            Command output (stdout and stderr)
        """
        import subprocess

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=60,
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += result.stderr
            if result.returncode != 0:
                output += f"\n\nExit code: {result.returncode}"

            return output if output else "(no output)"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 60 seconds"
        except Exception as e:
            return f"Error: {e}"

    return [ls, read_file, write_file, edit_file, glob, grep, execute]
