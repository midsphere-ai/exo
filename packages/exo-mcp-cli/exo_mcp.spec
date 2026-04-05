# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building a standalone exo-mcp binary.

Build:
    cd packages/exo-mcp-cli
    pyinstaller exo_mcp.spec --clean

Output:
    dist/exo-mcp
"""

a = Analysis(
    ["src/exo_mcp_cli/main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "mcp.client.stdio",
        "mcp.client.sse",
        "mcp.client.streamable_http",
        "mcp.client.websocket",
        "mcp.server.fastmcp",
        "typer",
        "rich",
        "cryptography",
        "cryptography.fernet",
        "cryptography.hazmat.primitives.hashes",
        "cryptography.hazmat.primitives.kdf.pbkdf2",
        "pydantic",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "exo",
        "exo_cli",
        "exo_web",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="exo-mcp",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
