# Third-Party Notices

This file summarizes third-party components currently identified in this repository for redistribution clarity.
It is an informational attribution file and does not replace the original license texts shipped with each component.

## 1. stb image libraries

Repository paths:
- `3rdParty/stb/stb_image.h`
- `3rdParty/stb/stb_image_resize2.h`

Upstream project:
- <https://github.com/nothings/stb>

Observed license information:
- `stb_image.h` contains its own embedded permissive license notice from Sean Barrett.
- Please retain the embedded notice when redistributing the file.

## 2. googletest

Referenced via git submodule configuration:
- Path: `3rdParty/googletest`
- Upstream: <https://github.com/google/googletest>

This component is referenced from `.gitmodules` and may or may not be checked out in every distribution copy.
Please refer to the upstream repository for its complete license text and notices.

## 3. nlohmann/json

Referenced via git submodule configuration:
- Path: `3rdParty/nlohmannJson`
- Upstream: <https://github.com/nlohmann/json>

This component is referenced from `.gitmodules` and may or may not be checked out in every distribution copy.
Please refer to the upstream repository for its complete license text and notices.

## Notes

- The primary project license remains Apache License 2.0.
- Third-party components may use different permissive licenses.
- Original notices embedded in source files or included by upstream dependencies must be preserved.
