"""Patch DeepCode (deepcode-hku) package issues after pip install.

Fixes:
1. Missing modules: prompts/, config/, schema/ not bundled in PyPI wheel.
2. Missing Ollama provider support in llm_utils.py.

Run via: ml-research setup deepcode
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def get_site_packages() -> Path:
    """Find the site-packages directory of the current venv."""
    for p in sys.path:
        if p.endswith("site-packages") and "deepcode" in str(Path(p).parent):
            return Path(p)
    # Fallback: use first site-packages
    for p in sys.path:
        if "site-packages" in p:
            return Path(p)
    raise RuntimeError("Cannot find site-packages directory")


REPO_BASE = "https://raw.githubusercontent.com/AyaraOL/https-github.com-HKUDS-DeepCode/main"

MISSING_DIRS = {
    "prompts": ["code_prompts.py", "__init__.py"],
    "config": ["mcp_tool_definitions.py", "mcp_tool_definitions_index.py"],
    "schema": ["output_schema.py"],
}

OLLAMA_PATCH = '''
    elif provider == "ollama":
        from mcp_agent.workflows.llm.augmented_llm_ollama import OllamaAugmentedLLM

        return OllamaAugmentedLLM
    else:
'''

OLLAMA_PROVIDER_KEYS = '            "ollama": ("local", "OllamaAugmentedLLM"),  # Ollama needs no key\n'


def download_missing_modules(site: Path) -> list[str]:
    """Download missing DeepCode modules from GitHub."""
    import httpx

    fixed = []
    for dirname, files in MISSING_DIRS.items():
        target_dir = site / dirname
        target_dir.mkdir(exist_ok=True)
        # Ensure __init__.py exists
        init_file = target_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")

        for fname in files:
            target = target_dir / fname
            if target.exists():
                continue
            url = f"{REPO_BASE}/{dirname}/{fname}"
            try:
                r = httpx.get(url, timeout=15, follow_redirects=True)
                if r.status_code == 200:
                    target.write_text(r.text)
                    fixed.append(f"{dirname}/{fname}")
            except Exception as e:
                print(f"  ⚠️ Failed to download {dirname}/{fname}: {e}")

    return fixed


def patch_ollama_support(site: Path) -> list[str]:
    """Patch DeepCode's llm_utils.py to add Ollama provider support."""
    patched = []
    llm_utils = site / "utils" / "llm_utils.py"
    if not llm_utils.exists():
        return patched

    content = llm_utils.read_text()

    # 1. Add Ollama to _get_llm_class
    if 'elif provider == "ollama":' not in content:
        old = '    else:\n        raise ValueError(f"Unknown provider: {provider}")'
        new = '    elif provider == "ollama":\n        from mcp_agent.workflows.llm.augmented_llm_ollama import OllamaAugmentedLLM\n\n        return OllamaAugmentedLLM\n    else:\n        raise ValueError(f"Unknown provider: {provider}")'
        content = content.replace(old, new)
        patched.append("_get_llm_class: +ollama")

    # 2. Add Ollama to provider_keys in get_preferred_llm_class
    if '"ollama"' not in content:
        old = '            "openai": (openai_key, "OpenAIAugmentedLLM"),\n        }'
        new = '            "openai": (openai_key, "OpenAIAugmentedLLM"),\n            "ollama": ("local", "OllamaAugmentedLLM"),  # Ollama needs no key\n        }'
        content = content.replace(old, new)
        patched.append("get_preferred_llm_class: +ollama provider")

    if patched:
        llm_utils.write_text(content)

    return patched


def run_setup() -> None:
    """Run all DeepCode patches."""
    print("🔧 DeepCode Package Patcher")
    print("=" * 40)

    site = get_site_packages()
    print(f"📍 Site-packages: {site}")

    # Check deepcode is installed
    if not (site / "workflows").exists():
        print("❌ DeepCode (deepcode-hku) not found in site-packages")
        print("   Install first: uv add deepcode-hku")
        return

    # 1. Download missing modules
    print("\n📦 Checking missing modules...")
    fixed = download_missing_modules(site)
    if fixed:
        for f in fixed:
            print(f"  ✅ Downloaded: {f}")
    else:
        print("  ✅ All modules present")

    # 2. Patch Ollama support
    print("\n🦙 Patching Ollama support...")
    patched = patch_ollama_support(site)
    if patched:
        for p in patched:
            print(f"  ✅ Patched: {p}")
    else:
        print("  ✅ Ollama already supported")

    # 3. Verify
    print("\n🔍 Verification...")
    try:
        from prompts.code_prompts import PAPER_INPUT_ANALYZER_PROMPT

        print(f"  ✅ prompts loaded ({len(PAPER_INPUT_ANALYZER_PROMPT)} chars)")
    except ImportError as e:
        print(f"  ❌ prompts import failed: {e}")

    try:
        from utils.llm_utils import get_preferred_llm_class

        print("  ✅ llm_utils accessible")
    except ImportError as e:
        print(f"  ❌ llm_utils import failed: {e}")

    print("\n✅ DeepCode setup complete!")


if __name__ == "__main__":
    run_setup()
