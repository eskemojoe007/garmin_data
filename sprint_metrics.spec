# -*- mode: python -*-

block_cipher = None


a = Analysis(['sprint_metrics.py'],
             pathex=['C:\\Users\\212333077\\Documents\\GitHub\\garmin_data'],
             binaries=[],
             datas=[],
             hiddenimports=['pandas._libs.tslibs.timedeltas','FileDialog', 'Tkinter'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='sprint_metrics',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
