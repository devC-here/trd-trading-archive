# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['MAIN.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('candle_ratios_model_best.pth', '.'),
        ('candle_ratios_scaler.pkl', '.'),
    ],
    hiddenimports=[
        'classifier',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MAIN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    icon='my_icon.ico',
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MAIN',
)