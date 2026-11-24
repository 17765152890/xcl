---
name: xcl-input
description: >-
  Change gamepad and keyboard input in XCL. Use when a title sees no controller,
  GameInputCreate fails, or when editing hle/gameinput.rs or hle/kbm.rs.
---

# Input

`hle/gameinput.rs` exports `GameInputCreate`. The console DLL imports it by ordinal 1 (`stubdb.toml`). The PC's XInput controller is one gamepad. Hot-plug is required: connect and disconnect while the process runs.

`hle/kbm.rs` feeds keyboard and mouse as that gamepad: WASD left stick, mouse right stick, mouse buttons triggers, wheel bumpers, Enter/Space A, Escape/Backspace B, arrows D-pad, E/Q Y/X, Shift/Ctrl stick clicks, Tab View, P Menu. `xcl.toml` `keyboard_mouse` and `mouse_sensitivity` disable it or scale the mouse. Both devices may be active together.

`gameinput.dll` module default in `stubdb.toml` is `E_NOTIMPL`. A new export needs a real implementation or the title will treat a fake success as a live device. Test with `tests/testtitle` input checks, not a retail exe.
