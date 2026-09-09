# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['MAIN.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('breakout_model.pth', '.'),
        ('breakout_scaler.pkl', '.'),
        ('candle_seq_model.pth', '.'),
        ('candle_seq_scaler.pkl', '.'),
        ('fusion_trade_log.json', '.'),
        ('microstructure_model.pth', '.'),
        ('microstructure_scaler.pkl', '.'),
        ('momentum_model.pth', '.'),
        ('momentum_scaler.pkl', '.'),
        ('vwap_volume_model.pth', '.'),
        ('vwap_volume_scaler.pkl', '.'),
        ('my_icon.ico', '.'),
    ],
    hiddenimports=['sklearn'], # ADDED 'sklearn' HERE
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
    a.binaries,
    a.datas, # 'a.datas' contains all the data files, including the icon
    [],
    name='MAIN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True, # Set to False if it's a GUI application
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # The 'icon' argument here refers to the *absolute path* to the icon
    # or a path relative to the .spec file. Since it's now in 'a.datas',
    # you can just refer to its name.
    icon=['my_icon.ico'],
)

# This part is for one-directory builds. For --onefile, the EXE object is typically the final target.
# If you are strictly using --onefile, you might not need a COLLECT section or it might be set up differently.
# However, if your original build was generating a folder, this section would be critical.
# Assuming you want a single file .exe, the above 'exe' definition is sufficient for the bundling.
# If the goal is a *single file* .exe, ensure no 'COLLECT' block is implicitly creating a directory,
# or that 'COLLECT' is configured for onefile output if generated this way.
# For a pure --onefile spec, usually, the generated spec file itself will handle this.