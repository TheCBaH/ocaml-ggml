# Plan: Generate OCaml Ctypes Bindings for gpt-2-lib.h

This plan outlines the steps to generate OCaml ctypes bindings for the C library defined in `src/gpt-2-lib.h`, following the conventions observed in the `lib/` directory and integrating with the existing build setup in `src/dune`.

## Steps

1.  **Define OCaml Types:**
    *   Create `src/gpt2_types.mli`: Define abstract OCaml types for `gpt2_model_t` and `ggml_cgraph`.
    *   Create `src/gpt2_types.ml`: Provide the corresponding empty implementation file.
2.  **Define Ctypes Type Descriptions:**
    *   Create `src/gpt2_type_description.ml`:
        *   Define a functor `Types(F : Ctypes.TYPE)`.
        *   Inside, define `ctypes` structure descriptions for `gpt2_model_t` (containing a pointer to the opaque `gpt2_model_buf`) and `ggml_cgraph` (as an opaque struct pointer). Reference the abstract types from `Gpt2_types`.
3.  **Define Ctypes Function Bindings:**
    *   Create `src/gpt2_function_description.ml`:
        *   Define a functor `Functions(F : Ctypes.FOREIGN)`.
        *   Inside, use `F.foreign` to bind the C functions: `ggml_main`, `gpt2_model_init`, `gpt2_model_uninit`, `gpt2_model_graph`. Use the generated types (e.g., `Gpt2_types_generated.gpt2_model_t`).
4.  **Update `src/dune`:**
    *   Modify the `(library ...)` stanza with `(name gpt-2)`.
    *   Add `gpt2_types` to the `(libraries ...)` list.
    *   Update the `(ctypes ...)` stanza:
        *   Change `(type_description ...)` to use `(instance Gpt2_types)` and `(functor Gpt2_type_description)`.
        *   Change `(function_description ...)` to use `(instance Gpt2_functions)` and `(functor Gpt2_function_description)`.
        *   Update generated file names if desired, e.g., `(generated_types Gpt2_types_generated)` and `(generated_entry_point Gpt2_c)`.
5.  **Build Verification:**
    *   Run `make` to verify the C library compilation (which implicitly triggers the OCaml build via the dune rule).

## Visualization

```mermaid
graph TD
    subgraph Preparation
        A[Analyze src/gpt-2-lib.h]
        B[Analyze lib/* convention files]
        C[Analyze src/dune build setup]
    end

    subgraph Implementation
        D[Create src/gpt2_types.mli/.ml] --> F[Define OCaml types for gpt2_model_t, ggml_cgraph]
        E[Create src/gpt2_type_description.ml] --> G[Define ctypes descriptions using functor pattern]
        H[Create src/gpt2_function_description.ml] --> I[Define ctypes bindings using functor pattern]
        J[Modify src/dune] --> K[Update library stanza to use new types/descriptions/functors]
    end

    subgraph Verification
        L{Build} --> M[Run 'make']
        M -- Success --> N[Bindings Generated & Compiled]
        M -- Failure --> O[Debug Build Errors]
    end

    Preparation --> Implementation
    Implementation --> Verification