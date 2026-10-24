import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_COMMENT = (
    "# Critical pin pair: chromadb==1.0.7 requires posthog==3.25.0 "
    "(newer posthog may break telemetry init)."
)


class RequirementsFileTests(unittest.TestCase):
    def test_requirements_files_are_utf8_without_nulls(self):
        paths = sorted(ROOT.glob("requirements*.txt"))
        self.assertTrue(paths)
        for path in paths:
            data = path.read_bytes()
            text = data.decode("utf-8")
            self.assertNotIn("\x00", text, path.name)
            self.assertFalse(data.startswith(b"\xff\xfe"), path.name)
            self.assertFalse(data.startswith(b"\xfe\xff"), path.name)

    def test_requirements_txt_keeps_chromadb_posthog_pin(self):
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn(PIN_COMMENT, text)
        self.assertIn("chromadb==1.0.7\n", text)
        self.assertIn("posthog==3.25.0\n", text)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu\n", text)
        self.assertIn("torch==2.7.0 ; sys_platform == 'darwin'\n", text)
        self.assertIn("torch==2.7.0+cpu ; sys_platform != 'darwin'\n", text)

    def test_docker_installs_requirements_txt_only(self):
        self.assertFalse((ROOT / "requirements.docker").exists())
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("requirements.txt", dockerfile)
        self.assertNotIn("requirements.docker", dockerfile)

    def test_docker_build_excludes_env_file(self):
        ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        self.assertIn(".env", ignored)

    def test_gitignore_covers_secrets_and_local_envs(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        for entry in (".env", ".env.*", "!.env.example", ".venv/", "/venv", "/data/", ".streamlit/secrets.toml"):
            self.assertIn(entry, ignored)
