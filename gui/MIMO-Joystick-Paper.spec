# -*- mode: python ; coding: utf-8 -*-
# 论文实验版 EXE：在原 MIMO-Joystick 基础上增加论文 E1-E4 实验按钮
# 与数据采集按钮（tp13_k20_k30 冻结参数）。构建：
#   python -m PyInstaller MIMO-Joystick-Paper.spec --noconfirm


a = Analysis(
    ['joystick_car.py'],
    pathex=[],
    binaries=[],
    datas=[('mimo_car_studio', 'mimo_car_studio')],
    hiddenimports=['serial.tools.list_ports', 'matplotlib.backends.backend_tkagg', 'queue'],
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
    a.datas,
    [],
    name='MIMO-Joystick-Paper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
