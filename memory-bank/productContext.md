# Product Context

This file provides a high-level overview of the project and the expected product that will be created. Initially it is based upon projectBrief.md (if provided) and all other available project-related information in the working directory. This file is intended to be updated as the project evolves, and should be used to inform all other modes of the project's goals and context.
2025-04-17 04:52:24 - Log of updates made will be appended as footnotes to the end of this file.

*

## Project Goal

*   Generate OCaml ctypes bindings for `src/gpt-2-lib.h`.
*   Follow conventions from `lib/ggml`.
*   Verify build using `make`.

## Key Features

*   OCaml bindings for gpt-2 C library functions and types.

## Overall Architecture

*   Use OCaml `ctypes` library to interface with the C header.
*   Organize bindings similar to existing `lib/` structure.
*   Integrate with the existing `Makefile` or `dune` build system.